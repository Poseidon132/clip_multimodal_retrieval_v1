"""BLIP image-text matching reranker for CLIP Top-K candidates.

No Transformers import is performed until BlipITMScorer is constructed; all
ranking code can be unit-tested offline. This reranks *within* the CLIP Top-K,
so Recall@K for the same candidate cutoff is mathematically unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Highest scores first; deterministic index tie-break for selected set."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be finite, non-empty, and one-dimensional")
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(values))
    # A complete stable sort also gives deterministic behavior on cutoff ties.
    return np.argsort(-values, kind="stable")[:k]


def minmax_scores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    width = float(np.max(x) - np.min(x))
    if width < 1e-12:
        return np.full_like(x, 0.5)
    return (x - np.min(x)) / width


def rerank_top_k(
    candidate_ids: Sequence[int],
    clip_scores: Sequence[float],
    blip_match_prob: Sequence[float],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix normalized CLIP similarity and BLIP matching probabilities.

    alpha=0 preserves the original CLIP candidate order. alpha=1 selects BLIP.
    """
    ids = np.asarray(candidate_ids, dtype=np.int64)
    clip = np.asarray(clip_scores, dtype=np.float64)
    blip = np.asarray(blip_match_prob, dtype=np.float64)
    if not (len(ids) and len(ids) == len(clip) == len(blip)):
        raise ValueError("candidate IDs / CLIP scores / BLIP scores must have equal nonzero lengths")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie within [0,1]")
    if not (np.isfinite(clip).all() and np.isfinite(blip).all()):
        raise ValueError("Non-finite scores")
    if np.any((blip < 0) | (blip > 1)):
        raise ValueError("BLIP matching probabilities must be between 0 and 1")
    combined = (1.0 - alpha) * minmax_scores(clip) + alpha * blip
    # Use the incoming CLIP order for stable ties.
    order = np.argsort(-combined, kind="stable")
    return ids[order], combined[order]


def _move_processor_inputs(inputs, device: torch.device, precision: torch.dtype):
    return {
        key: val.to(device=device, dtype=precision if val.is_floating_point() else None)
        for key, val in inputs.items()
    }


class BlipITMScorer:
    """Image-text matching probability from the BLIP ITM classification head."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        device: str | None = None,
        precision: str = "auto",
        checkpoint: str | None = None,
        model=None,
        processor=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision not in {"auto", "fp16", "fp32"}:
            raise ValueError("precision must be one of auto/fp16/fp32")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 BLIP inference requires CUDA")
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be provided together")
        if model is None:
            from transformers import AutoProcessor, BlipForImageTextRetrieval
            self.dtype = torch.float16 if (self.device.type == "cuda" and precision != "fp32") else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name)
            try:
                model = BlipForImageTextRetrieval.from_pretrained(model_name, dtype=self.dtype)
            except TypeError:
                model = BlipForImageTextRetrieval.from_pretrained(model_name, torch_dtype=self.dtype)
            if checkpoint:
                from .blip_itm_training import load_blip_itm_adapter
                info = load_blip_itm_adapter(model, checkpoint, base_model=model_name)
                print(f"BLIP ITM adapter loaded: epoch={info['epoch']} | val Mean R@1={info['validation_mean_R@1']:.4f}")
            self.model = model.to(self.device).eval()
            self.use_bf16_autocast = False
        else:
            if checkpoint:
                raise ValueError("Do not load checkpoint into an externally provided training model")
            self.model = model
            self.processor = processor
            self.dtype = next(model.parameters()).dtype
            # Training keeps parameters in FP32; run validation efficiently with BF16 autocast.
            self.use_bf16_autocast = (
                self.device.type == "cuda" and self.dtype == torch.float32
                and torch.cuda.is_bf16_supported()
            )
            self.model.eval()
        self.model_name = model_name

    @torch.inference_mode()
    def score_pairs(
        self,
        images: Sequence[str | Path | Image.Image],
        captions: Sequence[str],
        batch_size: int = 4,
    ) -> np.ndarray:
        if len(images) != len(captions):
            raise ValueError("Need one image and one caption per BLIP pair")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        outputs = []
        for start in range(0, len(images), batch_size):
            batch_images = []
            for image in images[start:start + batch_size]:
                if isinstance(image, Image.Image):
                    batch_images.append(image.convert("RGB"))
                else:
                    with Image.open(image) as source:
                        batch_images.append(source.convert("RGB"))
            try:
                inputs = self.processor(
                    images=batch_images,
                    text=list(captions[start:start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=35,
                    return_tensors="pt",
                )
            finally:
                for img in batch_images:
                    img.close()
            inputs = _move_processor_inputs(inputs, self.device, self.dtype)
            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=getattr(self, "use_bf16_autocast", False),
            ):
                result = self.model(**inputs, use_itm_head=True)
            logits = result.itm_score if hasattr(result, "itm_score") else result[0]
            if logits.ndim != 2 or logits.shape[-1] != 2:
                raise ValueError(f"Expected BLIP ITM [batch, 2] logits, got {tuple(logits.shape)}")
            probs = logits.float().softmax(dim=-1)[:, 1]
            outputs.append(probs.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0,), dtype="float32")
