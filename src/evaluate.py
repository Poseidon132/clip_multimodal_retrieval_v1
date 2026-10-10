from __future__ import annotations

import argparse
from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import load_caption_table
from .metrics import retrieval_metrics
from .model import ClipRetriever
from .utils import get_device, save_json


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.inference_mode()
def evaluate_model(
    model: ClipRetriever,
    csv_path: str,
    image_root: str,
    split: str,
    batch_size: int = 32,
    show_progress: bool = False,
) -> dict[str, float]:
    """Evaluate an already-loaded CLIP model without reloading weights.

    The same scoring and positive-caption rules are used by training-time
    validation and the standalone evaluation CLI.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        df = load_caption_table(csv_path)
        df = df[df["split"] == split.lower()].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with split={split} in {csv_path}")

        image_names = list(dict.fromkeys(df["image"].tolist()))
        image_to_idx = {name: i for i, name in enumerate(image_names)}

        root = Path(image_root)
        image_features = []
        image_batches = batched(image_names, batch_size)
        for names in tqdm(
            image_batches,
            total=ceil(len(image_names) / batch_size),
            desc="images",
            disable=not show_progress,
        ):
            images = []
            for name in names:
                with Image.open(root / name) as img:
                    images.append(img.convert("RGB"))
            embedding = model.encode_images(images, device)
            image_features.append(embedding.float().cpu().numpy())
        image_features = np.concatenate(image_features, axis=0)

        texts = df["caption"].tolist()
        text_features = []
        text_batch_size = batch_size * 2
        for batch_texts in tqdm(
            batched(texts, text_batch_size),
            total=ceil(len(texts) / text_batch_size),
            desc="texts",
            disable=not show_progress,
        ):
            embedding = model.encode_texts(batch_texts, device)
            text_features.append(embedding.float().cpu().numpy())
        text_features = np.concatenate(text_features, axis=0)

        similarity = text_features @ image_features.T
        target_image_indices = np.array(
            [image_to_idx[name] for name in df["image"].tolist()], dtype=np.int64
        )
        positives = defaultdict(list)
        for text_idx, image_idx in enumerate(target_image_indices.tolist()):
            positives[image_idx].append(text_idx)
        positive_text_indices = [positives[i] for i in range(len(image_names))]

        metrics = retrieval_metrics(similarity, target_image_indices, positive_text_indices)
        metrics["num_images"] = len(image_names)
        metrics["num_captions"] = len(texts)
        return metrics
    finally:
        if was_training:
            model.train()


def evaluate(
    csv_path: str,
    image_root: str,
    split: str,
    model_name: str,
    adapter: str | None,
    batch_size: int,
    output: str,
    adapter_mode: str = "both",
) -> dict[str, float]:
    device = get_device()
    model = ClipRetriever(model_name=model_name, mode="frozen").to(device)
    if adapter:
        model.load_adapter(adapter, adapter_mode=adapter_mode)
    elif adapter_mode != "both":
        raise ValueError("image-only requires --adapter")
    metrics = evaluate_model(
        model, csv_path, image_root, split, batch_size, show_progress=True
    )
    save_json(metrics, output)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/toy/captions.csv")
    p.add_argument("--image-root", default="data/toy/images")
    p.add_argument("--split", default="test")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--adapter", default=None)
    p.add_argument("--adapter-mode", choices=["both", "image-only"], default="both")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="artifacts/eval_metrics.json")
    args = p.parse_args()
    evaluate(args.csv, args.image_root, args.split, args.model_name, args.adapter, args.batch_size, args.output, args.adapter_mode)


if __name__ == "__main__":
    main()
