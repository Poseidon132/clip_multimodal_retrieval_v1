"""Single-query CLIP retrieval latency and GPU allocation (val/test gallery).

Records warm single-query p50/p95, including image/text preprocessing,
encoder forward, and vector index search. Excludes one-time model loading.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_caption_table
from src.search import ExactMultimodalIndex
from src.utils import get_device, save_json
from scripts.eval_blip_rerank import sampled_query_indices, subset_gallery


def summarize_ms(seconds: list[float]) -> dict:
    values = np.array(seconds, dtype=np.float64) * 1000.0
    return {"median_ms": round(float(np.median(values)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3),
            "mean_ms": round(float(np.mean(values)), 3)}


def run(args):
    from src.model import ClipRetriever

    idx = ExactMultimodalIndex(args.index)
    idx.validate_encoder(args.model_name, args.adapter, args.adapter_mode)
    gallery = subset_gallery(idx, args.csv, args.split)
    tq = sampled_query_indices(len(gallery["captions"]), args.sample_text_queries, args.seed)
    iq = sampled_query_indices(len(gallery["image_names"]), args.sample_image_queries, args.seed + 1)
    device = get_device()
    model = ClipRetriever(args.model_name, "frozen").to(device).eval()
    if args.adapter:
        model.load_adapter(args.adapter, adapter_mode=args.adapter_mode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    durations_t, durations_i = [], []
    with torch.inference_mode():
        for id in tqdm(tq, desc="CLIP text query"):
            text = str(gallery["captions"][int(id)])
            sync(); t = time.perf_counter()
            query = model.encode_texts([text], device).float().cpu().numpy()[0]
            _ = gallery["image_features"] @ query
            sync(); durations_t.append(time.perf_counter() - t)
        for id in tqdm(iq, desc="CLIP image query"):
            name = str(gallery["image_names"][int(id)])
            sync(); t = time.perf_counter()
            with Image.open(Path(args.image_root) / name) as img:
                image = img.convert("RGB")
            query = model.encode_images([image], device).float().cpu().numpy()[0]
            _ = gallery["text_features"] @ query
            sync(); durations_i.append(time.perf_counter() - t)
            image.close()
    result = {
        "split": args.split, "num_images": len(gallery["image_names"]),
        "num_captions": len(gallery["captions"]),
        "evaluated_text_queries": len(tq), "evaluated_image_queries": len(iq),
        "latency_includes": "CPU image/text preprocess + CLIP encoder + float32 cosine dot; excludes model loading",
        "text_to_image": summarize_ms(durations_t),
        "image_to_text": summarize_ms(durations_i),
        "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3) if device.type == "cuda" else 0.0,
    }
    save_json(result, args.output)
    print(f"CLIP T->I median={result['text_to_image']['median_ms']:.2f}ms, I->T median={result['image_to_text']['median_ms']:.2f}ms, peak={result['gpu_peak_allocated_gb']:.2f}GB")
    print(f"Saved: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/flickr8k/captions_official.csv")
    p.add_argument("--image-root", default="data/flickr8k/Images")
    p.add_argument("--index", default="artifacts/flickr8k_asymmetric_index.npz")
    p.add_argument("--adapter", default="artifacts/flickr8k_best_hn02/adapter.pt")
    p.add_argument("--adapter-mode", choices=("image-only", "both"), default="image-only")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--sample-text-queries", type=int, default=100)
    p.add_argument("--sample-image-queries", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/clip_val_query_latency.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
