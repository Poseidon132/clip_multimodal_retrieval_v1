"""Two-stage CLIP candidate retrieval + BLIP ITM Top-K reranking.

Usage (from project root): python scripts/eval_blip_rerank.py --split val ...

The gallery ALWAYS remains the full val/test split, even in --sample-* mode.
Only the number of evaluated queries changes; query subsets are seeded.
BLIP can reorder CLIP Top-K, never rescue an item below CLIP Top-K.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blip_rerank import BlipITMScorer, rerank_top_k, top_k_indices
from src.dataset import load_caption_table
from src.search import ExactMultimodalIndex
from src.utils import save_json


def sampled_query_indices(total: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count < 0:
        raise ValueError("sample size must be >= 0; use 0 to evaluate all queries")
    if sample_count == 0 or sample_count >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size=sample_count, replace=False))


def subset_gallery(index: ExactMultimodalIndex, csv_path: str, split: str) -> dict:
    """Use only official split examples, verifying embedding/caption alignment."""
    df = load_caption_table(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No examples for split={split!r}")
    names = list(dict.fromkeys(df["image"].tolist()))
    name_set = set(names)
    img_mask = np.isin(index.image_names, list(name_set))
    text_mask = np.isin(index.caption_images, list(name_set))
    gallery_image_names = index.image_names[img_mask]
    gallery_texts = index.captions[text_mask]
    gallery_text_images = index.caption_images[text_mask]
    if len(gallery_image_names) != len(names) or set(gallery_image_names) != name_set:
        raise ValueError("The feature index is missing images; rebuild it using official captions CSV")
    if not np.array_equal(gallery_text_images, df["image"].to_numpy(dtype=str)) or not np.array_equal(
        gallery_texts, df["caption"].to_numpy(dtype=str)
    ):
        raise ValueError("Index captions do not match this CSV and split. Rebuild the index with this CSV.")
    image_to_idx = {name: i for i, name in enumerate(gallery_image_names)}
    targets = np.array([image_to_idx[name] for name in gallery_text_images], dtype=np.int64)
    positives = [[] for _ in range(len(gallery_image_names))]
    for text_id, image_id in enumerate(targets):
        positives[int(image_id)].append(text_id)
    return {
        "image_names": gallery_image_names,
        "image_features": index.image_features[img_mask],
        "captions": gallery_texts,
        "text_features": index.text_features[text_mask],
        "targets": targets,
        "positive_texts": positives,
    }


def evaluate_ranked_candidate_lists(
    ranks_t2i: list[np.ndarray],
    ranks_i2t: list[np.ndarray],
    text_query_ids: np.ndarray,
    image_query_ids: np.ndarray,
    targets: np.ndarray,
    positive_texts: list[list[int]],
) -> dict[str, float]:
    """Recall@K on QUERY SUBSETS, with all gallery items still eligible."""
    result = {}
    if not len(ranks_t2i) or not len(ranks_i2t):
        raise ValueError("Both retrieval directions require >= 1 query")
    for k in (1, 5, 10):
        result[f"t2i_R@{k}"] = float(np.mean([
            int(targets[t]) in order[:k] for t, order in zip(text_query_ids, ranks_t2i)
        ]))
        result[f"i2t_R@{k}"] = float(np.mean([
            bool(set(positive_texts[int(i)]).intersection(order[:k].tolist()))
            for i, order in zip(image_query_ids, ranks_i2t)
        ]))
    result["mean_R@1"] = 0.5 * (result["t2i_R@1"] + result["i2t_R@1"])
    return result



def candidate_oracle_hit_rates(
    t2i_candidates, i2t_candidates, tq, iq, targets, positive_texts
) -> dict[str, float]:
    """The maximum attainable R@1 within the fixed CLIP Top-K candidate lists.

    This is NOT a model accuracy score: it assumes an ideal reranker that always
    selects a positive whenever the CLIP candidate list contains one.
    """
    t_hits = [
        int(targets[int(t)]) in ids
        for t, (ids, _) in zip(tq, t2i_candidates)
    ]
    i_hits = [
        bool(set(positive_texts[int(i)]).intersection(ids.tolist()))
        for i, (ids, _) in zip(iq, i2t_candidates)
    ]
    t_rate = float(np.mean(t_hits))
    i_rate = float(np.mean(i_hits))
    return {
        "t2i_candidate_hit_rate": t_rate,
        "i2t_candidate_hit_rate": i_rate,
        "mean_candidate_hit_rate": 0.5 * (t_rate + i_rate),
    }


def build_candidates(sim: np.ndarray, tq: np.ndarray, iq: np.ndarray, top_k: int):
    t2i_candidates, i2t_candidates = [], []
    for text_idx in tq:
        ids = top_k_indices(sim[int(text_idx)], top_k)
        t2i_candidates.append((ids, sim[int(text_idx), ids].copy()))
    for image_idx in iq:
        ids = top_k_indices(sim[:, int(image_idx)], top_k)
        i2t_candidates.append((ids, sim[ids, int(image_idx)].copy()))
    return t2i_candidates, i2t_candidates


def compute_metrics_for_alpha(
    t2i_candidates, i2t_candidates, t2i_blip, i2t_blip,
    alpha, tq, iq, targets, positive_texts,
):
    text_ranked = [
        rerank_top_k(ids, cosines, probs, alpha)[0]
        for (ids, cosines), probs in zip(t2i_candidates, t2i_blip)
    ]
    image_ranked = [
        rerank_top_k(ids, cosines, probs, alpha)[0]
        for (ids, cosines), probs in zip(i2t_candidates, i2t_blip)
    ]
    return evaluate_ranked_candidate_lists(text_ranked, image_ranked, tq, iq, targets, positive_texts)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(args, scorer=None):
    if args.top_k < 10:
        raise ValueError("Use --top-k >= 10 to report Recall@1, @5, @10 without truncated ranks")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.split == "test" and args.alpha is None:
        raise ValueError("Test evaluation requires a FIXED --alpha selected on validation")
    if args.split == "test" and (args.sample_text_queries or args.sample_image_queries):
        raise ValueError("Test evaluation must include all queries; set --sample-* 0")
    alphas = [float(args.alpha)] if args.alpha is not None else sorted(set(args.alphas))
    if any(a < 0 or a > 1 for a in alphas):
        raise ValueError("--alpha / --alphas must be in [0,1]")

    index = ExactMultimodalIndex(args.index)
    index.validate_encoder(args.clip_model, args.adapter, args.adapter_mode)
    gallery = subset_gallery(index, args.csv, args.split)
    n_images = len(gallery["image_names"])
    n_texts = len(gallery["captions"])
    tq = sampled_query_indices(n_texts, args.sample_text_queries, args.seed)
    iq = sampled_query_indices(n_images, args.sample_image_queries, args.seed + 1)

    print(f"Split={args.split} | gallery={n_images} images / {n_texts} captions | queries={len(tq)} text / {len(iq)} image")
    print(f"Top-{args.top_k} candidates | BLIP batch={args.batch_size} | alpha={alphas}")
    start = time.perf_counter()
    similarity = gallery["text_features"] @ gallery["image_features"].T
    t2i, i2t = build_candidates(similarity, tq, iq, args.top_k)
    retrieval_seconds = time.perf_counter() - start

    # Reference top-K predictions should reproduce the existing CLIP R@K.
    base = evaluate_ranked_candidate_lists(
        [ids for ids, _ in t2i], [ids for ids, _ in i2t],
        tq, iq, gallery["targets"], gallery["positive_texts"],
    )
    oracle = candidate_oracle_hit_rates(
        t2i, i2t, tq, iq, gallery["targets"], gallery["positive_texts"],
    )

    if scorer is None and any(a > 0 for a in alphas):
        print("Loading BLIP ITM (first run downloads ~900 MB of weights)...")
        scorer = BlipITMScorer(args.blip_model, precision=args.precision)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    root = Path(args.image_root)
    scores_t, scores_i = [], []
    _cuda_sync()
    start_blip = time.perf_counter()
    if scorer is not None:
        for text_id, (ids, _) in tqdm(zip(tq, t2i), total=len(tq), desc="BLIP T->I"):
            images = [root / gallery["image_names"][i] for i in ids]
            captions = [str(gallery["captions"][int(text_id)])] * len(ids)
            scores_t.append(scorer.score_pairs(images, captions, args.batch_size))
        _cuda_sync()
        text_seconds = time.perf_counter() - start_blip
        image_start = time.perf_counter()
        for image_id, (ids, _) in tqdm(zip(iq, i2t), total=len(iq), desc="BLIP I->T"):
            image = root / gallery["image_names"][int(image_id)]
            images = [image] * len(ids)
            captions = [str(gallery["captions"][i]) for i in ids]
            scores_i.append(scorer.score_pairs(images, captions, args.batch_size))
        _cuda_sync()
        image_seconds = time.perf_counter() - image_start
    else:
        scores_t = [np.zeros(len(ids), dtype=np.float32) for ids, _ in t2i]
        scores_i = [np.zeros(len(ids), dtype=np.float32) for ids, _ in i2t]
        text_seconds = image_seconds = 0.0

    entries = []
    for alpha in alphas:
        metrics = compute_metrics_for_alpha(
            t2i, i2t, scores_t, scores_i, alpha, tq, iq,
            gallery["targets"], gallery["positive_texts"],
        )
        # R@10 is invariant ONLY when the entire reranked list is exactly Top-10.
        # With Top-20/30, an item from ranks 11..K can enter (or leave) Top-10.
        if args.top_k == 10:
            for direction in ("t2i", "i2t"):
                if abs(metrics[f"{direction}_R@10"] - base[f"{direction}_R@10"]) > 1e-12:
                    raise AssertionError("Within-Top10 reranking unexpectedly changed R@10")
        entries.append({"alpha": alpha, **metrics})
    entries.sort(key=lambda e: (-e["mean_R@1"], -e["t2i_R@1"], e["alpha"]))
    best = entries[0]
    timing = {
        "clip_gallery_similarity_seconds": round(retrieval_seconds, 3),
        "blip_t2i_seconds": round(text_seconds, 3),
        "blip_i2t_seconds": round(image_seconds, 3),
        "blip_t2i_seconds_per_query": round(text_seconds / len(tq), 4),
        "blip_i2t_seconds_per_query": round(image_seconds / len(iq), 4),
        "blip_pairs_per_second": round((len(tq) + len(iq)) * args.top_k / max(text_seconds + image_seconds, 1e-9), 2) if scorer is not None else None,
        "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if torch.cuda.is_available() else 0.0,
    }
    result = {
        "split": args.split,
        "gallery_images": n_images,
        "gallery_captions": n_texts,
        "num_images": n_images,
        "num_captions": n_texts,
        "evaluated_text_queries": len(tq),
        "evaluated_image_queries": len(iq),
        "query_seed": args.seed,
        "candidate_top_k": args.top_k,
        "clip_index": args.index,
        "adapter_mode": args.adapter_mode,
        "blip_model": args.blip_model,
        "clip_baseline": base,
        "candidate_oracle_at_top_k": oracle,
        "results": entries,
        "selected": best,
        "selection_rule": "mean_R@1 on validation" if args.split == "val" and args.alpha is None else "fixed alpha, no tuning",
        "timing": timing,
    }
    save_json(result, args.output)
    print(" alpha  T->I R@1  I->T R@1   Mean R@1")
    print(f"  CLIP  {base['t2i_R@1'] * 100:8.2f}% {base['i2t_R@1'] * 100:10.2f}% {base['mean_R@1'] * 100:10.2f}%")
    print(f"  Top-{args.top_k} candidate ceiling (ideal reranker): "
          f"T->I {oracle['t2i_candidate_hit_rate'] * 100:.2f}% | "
          f"I->T {oracle['i2t_candidate_hit_rate'] * 100:.2f}% | "
          f"Mean {oracle['mean_candidate_hit_rate'] * 100:.2f}%")
    for row in entries:
        print(f" {row['alpha']:>5.2f} {row['t2i_R@1'] * 100:8.2f}% {row['i2t_R@1'] * 100:10.2f}% {row['mean_R@1'] * 100:10.2f}%")
    print(f"Timing: T->I {timing['blip_t2i_seconds_per_query']:.3f}s/query; I->T {timing['blip_i2t_seconds_per_query']:.3f}s/query; GPU peak={timing['gpu_peak_allocated_gb']:.2f}GB")
    print(f"Saved: {args.output}")
    if args.split == "val":
        print("Validation only. For the held-out test set, lock --alpha from FULL val (not a small sample).")
    return result


def main():
    p = argparse.ArgumentParser(description="CLIP Top-10 retrieval followed by BLIP ITM reranking")
    p.add_argument("--csv", default="data/flickr8k/captions_official.csv")
    p.add_argument("--image-root", default="data/flickr8k/Images")
    p.add_argument("--index", default="artifacts/flickr8k_asymmetric_index.npz")
    p.add_argument("--adapter", default="artifacts/flickr8k_best_hn02/adapter.pt")
    p.add_argument("--adapter-mode", default="image-only", choices=("image-only", "both"))
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--blip-model", default="Salesforce/blip-itm-base-coco")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    p.add_argument("--sample-text-queries", type=int, default=100, help="0 = ALL; on test both must be 0")
    p.add_argument("--sample-image-queries", type=int, default=20, help="0 = ALL; on test both must be 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", nargs="+", type=float, default=[0., 0.25, 0.5, 0.75, 1.])
    p.add_argument("--alpha", type=float, default=None, help="FIXED coefficient. Required for test.")
    p.add_argument("--output", default="artifacts/blip_val_small.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
