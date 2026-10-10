from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
def evaluate(csv_path: str, image_root: str, split: str, model_name: str, adapter: str | None, batch_size: int, output: str) -> dict[str, float]:
    device = get_device()
    model = ClipRetriever(model_name=model_name, mode="frozen").to(device).eval()
    if adapter:
        model.load_adapter(adapter)

    df = load_caption_table(csv_path)
    df = df[df["split"] == split.lower()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows with split={split}")

    image_names = list(dict.fromkeys(df["image"].tolist()))
    image_to_idx = {name: i for i, name in enumerate(image_names)}

    image_features = []
    for names in tqdm(list(batched(image_names, batch_size)), desc="images"):
        images = [Image.open(Path(image_root) / name).convert("RGB") for name in names]
        image_features.append(model.encode_images(images, device).cpu().numpy())
    image_features = np.concatenate(image_features, axis=0)

    texts = df["caption"].tolist()
    text_features = []
    for batch_texts in tqdm(list(batched(texts, batch_size * 2)), desc="texts"):
        text_features.append(model.encode_texts(batch_texts, device).cpu().numpy())
    text_features = np.concatenate(text_features, axis=0)

    similarity = text_features @ image_features.T
    target_image_indices = np.array([image_to_idx[n] for n in df["image"].tolist()], dtype=np.int64)
    positives = defaultdict(list)
    for text_idx, image_idx in enumerate(target_image_indices.tolist()):
        positives[image_idx].append(text_idx)
    positive_text_indices = [positives[i] for i in range(len(image_names))]

    metrics = retrieval_metrics(similarity, target_image_indices, positive_text_indices)
    metrics["num_images"] = len(image_names)
    metrics["num_captions"] = len(texts)
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="artifacts/eval_metrics.json")
    args = p.parse_args()
    evaluate(args.csv, args.image_root, args.split, args.model_name, args.adapter, args.batch_size, args.output)


if __name__ == "__main__":
    main()
