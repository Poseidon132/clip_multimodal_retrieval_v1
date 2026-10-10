"""Validation-first interpolation of original and fine-tuned CLIP embeddings.

Run from the project root with ``python scripts/eval_feature_blend.py ...``.
This is post-hoc feature interpolation, NOT a trained residual adapter.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Allow running `python scripts/eval_feature_blend.py` directly on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_caption_table
from src.metrics import retrieval_metrics
from src.model import ClipRetriever
from src.utils import get_device, save_json


def _batches(items, n):
    for offset in range(0, len(items), n):
        yield items[offset : offset + n]


@torch.inference_mode()
def extract_features(model, image_names, captions, image_root, batch_size, device, title):
    model.eval()
    image_features = []
    for names in tqdm(
        _batches(image_names, batch_size),
        total=ceil(len(image_names) / batch_size),
        desc=f"{title}: images",
    ):
        images = []
        for name in names:
            with Image.open(image_root / name) as im:
                images.append(im.convert("RGB"))
        image_features.append(model.encode_images(images, device).float().cpu())

    text_features = []
    for texts in tqdm(
        _batches(captions, batch_size * 2),
        total=ceil(len(captions) / (batch_size * 2)),
        desc=f"{title}: texts",
    ):
        text_features.append(model.encode_texts(texts, device).float().cpu())

    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def mix_normalize(baseline: torch.Tensor, adapted: torch.Tensor, weight: float):
    """Keep a=0 and a=1 exactly equal to each original embedding set."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Weight must be between 0 and 1: {weight}")
    return F.normalize(
        baseline * (1.0 - weight) + adapted * weight, p=2, dim=-1
    )


@torch.inference_mode()
def score_mix(
    base_images,
    base_texts,
    tuned_images,
    tuned_texts,
    image_alpha,
    text_alpha,
    targets,
    positive_text_indices,
    device,
):
    images = mix_normalize(base_images, tuned_images, image_alpha).to(device)
    texts = mix_normalize(base_texts, tuned_texts, text_alpha).to(device)
    similarity = (texts @ images.T).float().cpu().numpy()
    return retrieval_metrics(similarity, targets, positive_text_indices)


def run(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    fixed = args.image_alpha is not None or args.text_alpha is not None
    if fixed and (args.image_alpha is None or args.text_alpha is None):
        raise ValueError("Specify BOTH --image-alpha and --text-alpha")
    if not fixed and args.split != "val":
        raise ValueError("Grid search is restricted to val; specify fixed alphas for test")
    if not Path(args.adapter).is_file():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    df = load_caption_table(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No caption rows for split={args.split!r}")

    image_names = list(dict.fromkeys(df["image"].tolist()))
    texts = df["caption"].tolist()
    image_to_idx = {name: i for i, name in enumerate(image_names)}
    targets = np.array([image_to_idx[name] for name in df["image"]], dtype=np.int64)
    positives = defaultdict(list)
    for text_idx, image_idx in enumerate(targets):
        positives[int(image_idx)].append(text_idx)
    positive_text_indices = [positives[i] for i in range(len(image_names))]

    device = get_device()
    print(f"Split: {args.split} | Device: {device} | {len(image_names)} images | {len(texts)} captions")
    model = ClipRetriever(model_name=args.model_name, mode="frozen").to(device)
    root = Path(args.image_root)
    base_images, base_texts = extract_features(
        model, image_names, texts, root, args.batch_size, device, "zero-shot"
    )
    model.load_adapter(args.adapter)
    tuned_images, tuned_texts = extract_features(
        model, image_names, texts, root, args.batch_size, device, "fine-tuned"
    )
    # The embeddings are small (<20MB each for Flickr8k validation/test).
    # Keep them on GPU for the nine 5000x1000 similarity products.
    base_images, base_texts = base_images.to(device), base_texts.to(device)
    tuned_images, tuned_texts = tuned_images.to(device), tuned_texts.to(device)

    if fixed:
        pairs = [(float(args.image_alpha), float(args.text_alpha))]
    else:
        alphas = sorted(set(float(a) for a in args.alphas))
        if any(a < 0.0 or a > 1.0 for a in alphas):
            raise ValueError("--alphas values must be in [0, 1]")
        pairs = list(product(alphas, alphas))

    results = []
    for image_alpha, text_alpha in pairs:
        metrics = score_mix(
            base_images, base_texts, tuned_images, tuned_texts,
            image_alpha, text_alpha, targets, positive_text_indices, device,
        )
        entry = {
            "image_alpha": image_alpha,
            "text_alpha": text_alpha,
            **metrics,
        }
        results.append(entry)

    # Higher validation Mean R@1 is better; use a deterministic tie-break.
    results.sort(key=lambda r: (-r["mean_R@1"], -r["i2t_R@1"], r["image_alpha"], r["text_alpha"]))
    print("image_a text_a   T->I R@1   I->T R@1  Mean R@1")
    for r in results:
        print(
            f"{r['image_alpha']:>7.2f} {r['text_alpha']:>6.2f} "
            f"{100*r['t2i_R@1']:>10.2f}% "
            f"{100*r['i2t_R@1']:>10.2f}% "
            f"{100*r['mean_R@1']:>10.2f}%"
        )
    best = results[0]
    output = {
        "split": args.split,
        "selection_metric": "mean_R@1" if not fixed else None,
        "adapter": args.adapter,
        "num_images": len(image_names),
        "num_captions": len(texts),
        "selected": best,
        "results": results,
    }
    save_json(output, args.output)
    print(f"Saved: {args.output}")
    if args.split == "val" and not fixed:
        print(
            "Selected on VAL only. For one held-out test evaluation, run with "
            f"--split test --image-alpha {best['image_alpha']:.2f} "
            f"--text-alpha {best['text_alpha']:.2f}"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Zero-shot / fine-tuned CLIP feature interpolation")
    parser.add_argument("--csv", default="data/flickr8k/captions_official.csv")
    parser.add_argument("--image-root", default="data/flickr8k/Images")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--adapter", default="artifacts/flickr8k_best_hn02/adapter.pt")
    parser.add_argument("--split", choices=("val", "test", "train"), default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--image-alpha", type=float, default=None)
    parser.add_argument("--text-alpha", type=float, default=None)
    parser.add_argument("--output", default="artifacts/flickr8k_blend_val.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
