from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .index_metadata import validate_index_metadata
from .utils import get_device


class ExactMultimodalIndex:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            stored_keys = ("index_model_name", "index_adapter_mode", "index_adapter_sha256")
            if all(k in data.files for k in stored_keys):
                self.encoder_metadata = {k: str(data[k].item()) for k in stored_keys}
            elif any(k in data.files for k in stored_keys):
                raise ValueError("Index contains incomplete encoder metadata; rebuild it")
            else:
                self.encoder_metadata = None
            self.image_names = data["image_names"].astype(str)
            self.image_features = data["image_features"].astype("float32")
            self.captions = data["captions"].astype(str)
            self.caption_images = data["caption_images"].astype(str)
            self.text_features = data["text_features"].astype("float32")
        # All arrays are loaded before np.load() closes the file.

    def validate_encoder(self, model_name: str, adapter: str | None, adapter_mode: str = "both") -> None:
        validate_index_metadata(self.encoder_metadata, model_name, adapter, adapter_mode)

    @staticmethod
    def _topk(matrix: np.ndarray, query_feature: np.ndarray, top_k: int):
        scores = matrix @ query_feature.reshape(-1)
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

    def search_images(self, query_feature: np.ndarray, top_k: int = 5):
        idx, scores = self._topk(self.image_features, query_feature, top_k)
        return [(self.image_names[i], float(scores[j])) for j, i in enumerate(idx)]

    def search_captions(self, query_feature: np.ndarray, top_k: int = 5):
        idx, scores = self._topk(self.text_features, query_feature, top_k)
        return [
            (self.captions[i], self.caption_images[i], float(scores[j]))
            for j, i in enumerate(idx)
        ]


@torch.inference_mode()
def run_search(
    index_path: str,
    model_name: str,
    adapter: str | None,
    top_k: int,
    text_query: str | None,
    image_query: str | None,
    adapter_mode: str = "both",
):
    index = ExactMultimodalIndex(index_path)
    index.validate_encoder(model_name, adapter, adapter_mode)
    device = get_device()
    from .model import ClipRetriever
    model = ClipRetriever(model_name=model_name, mode="frozen").to(device).eval()
    if adapter:
        model.load_adapter(adapter, adapter_mode=adapter_mode)

    if text_query:
        q = model.encode_texts([text_query], device).cpu().numpy()[0]
        return "text_to_image", index.search_images(q, top_k)

    image = Image.open(image_query).convert("RGB")
    q = model.encode_images([image], device).cpu().numpy()[0]
    return "image_to_text", index.search_captions(q, top_k)


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", default=None)
    group.add_argument("--image", default=None)
    p.add_argument("--index", default="artifacts/image_text_index.npz")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--adapter", default=None)
    p.add_argument("--adapter-mode", choices=["both", "image-only"], default="both")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    mode, results = run_search(args.index, args.model_name, args.adapter, args.top_k, args.text, args.image, args.adapter_mode)
    if mode == "text_to_image":
        for rank, (name, score) in enumerate(results, 1):
            print(f"{rank:02d}. {name}  score={score:.4f}")
    else:
        for rank, (caption, image_name, score) in enumerate(results, 1):
            print(f"{rank:02d}. score={score:.4f}  source={image_name}  caption={caption}")


if __name__ == "__main__":
    main()
