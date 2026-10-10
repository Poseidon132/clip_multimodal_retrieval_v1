from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import load_caption_table
from .index_metadata import get_index_metadata
from .model import ClipRetriever
from .utils import get_device


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.inference_mode()
def build_index(csv_path: str, image_root: str, model_name: str, adapter: str | None, batch_size: int, output: str, adapter_mode: str = "both") -> None:
    metadata = get_index_metadata(model_name, adapter, adapter_mode)
    device = get_device()
    model = ClipRetriever(model_name=model_name, mode="frozen").to(device).eval()
    if adapter:
        model.load_adapter(adapter, adapter_mode=adapter_mode)

    df = load_caption_table(csv_path)
    image_names = list(dict.fromkeys(df["image"].tolist()))

    image_features = []
    for names in tqdm(list(batched(image_names, batch_size)), desc="image embeddings"):
        images = [Image.open(Path(image_root) / n).convert("RGB") for n in names]
        image_features.append(model.encode_images(images, device).cpu().numpy())
    image_features = np.concatenate(image_features, axis=0).astype("float32")

    captions = df["caption"].tolist()
    caption_images = df["image"].tolist()
    text_features = []
    for texts in tqdm(list(batched(captions, batch_size * 2)), desc="text embeddings"):
        text_features.append(model.encode_texts(texts, device).cpu().numpy())
    text_features = np.concatenate(text_features, axis=0).astype("float32")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_names=np.array(image_names),
        image_features=image_features,
        captions=np.array(captions),
        caption_images=np.array(caption_images),
        text_features=text_features,
        **metadata,
    )
    print(f"Saved {len(image_names)} image embeddings and {len(captions)} text embeddings to {output_path}")
    print(f"Index encoder: model={model_name}, adapter_mode={metadata['index_adapter_mode']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/toy/captions.csv")
    p.add_argument("--image-root", default="data/toy/images")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--adapter", default=None)
    p.add_argument("--adapter-mode", choices=["both", "image-only"], default="both")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="artifacts/image_text_index.npz")
    args = p.parse_args()
    build_index(args.csv, args.image_root, args.model_name, args.adapter, args.batch_size, args.output, args.adapter_mode)


if __name__ == "__main__":
    main()
