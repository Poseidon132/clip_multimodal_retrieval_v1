from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from .adapter_utils import apply_projection_adapter
from .utils import l2_normalize


class ClipRetriever(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", mode: str = "projection"):
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.set_train_mode(mode)

    def set_train_mode(self, mode: str) -> None:
        if mode not in {"projection", "visual-only", "full", "frozen"}:
            raise ValueError("mode must be one of: projection, visual-only, full, frozen")
        for p in self.clip.parameters():
            p.requires_grad = mode == "full"
        if mode == "projection":
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True
            for p in self.clip.text_projection.parameters():
                p.requires_grad = True
            self.clip.logit_scale.requires_grad = True
        if mode == "visual-only":
            # Preserve the pretrained text space: only the visual projection learns.
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True
            self.clip.logit_scale.requires_grad = False
        if mode == "frozen":
            self.clip.logit_scale.requires_grad = False
        self.mode = mode

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.logit_scale

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.clip.parameters() if p.requires_grad)

    @staticmethod
    def _feature_tensor(features) -> torch.Tensor:
        """Return the projected feature tensor across Transformers 4.x/5.x APIs.

        Transformers 4.x returns a Tensor from CLIPModel.get_*_features(), while
        Transformers 5.x returns BaseModelOutputWithPooling and stores the projected
        embedding in ``pooler_output``.
        """
        if isinstance(features, torch.Tensor):
            return features
        pooler_output = getattr(features, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        if isinstance(features, (tuple, list)) and features:
            # Defensive fallback for return_dict=False style outputs.
            for item in reversed(features):
                if isinstance(item, torch.Tensor) and item.ndim == 2:
                    return item
        raise TypeError(
            f"Unsupported CLIP feature output type: {type(features)!r}. "
            "Expected a Tensor or an object containing .pooler_output."
        )

    def encode_images(self, images, device: torch.device) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return l2_normalize(self._feature_tensor(features))

    def encode_texts(self, texts, device: torch.device) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = self.clip.get_text_features(**inputs)
        return l2_normalize(self._feature_tensor(features))

    def encode_pairs(self, images, texts, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(text=list(texts), images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.clip(**inputs)
        return l2_normalize(out.image_embeds), l2_normalize(out.text_embeds)

    def save_adapter(self, path: str | Path) -> None:
        state = {
            "model_name": self.model_name,
            "mode": self.mode,
            "visual_projection": self.clip.visual_projection.state_dict(),
            "text_projection": self.clip.text_projection.state_dict(),
            "logit_scale": self.clip.logit_scale.detach().cpu(),
        }
        if self.mode == "full":
            state["full_state_dict"] = self.clip.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_adapter(self, path: str | Path, adapter_mode: str = "both") -> None:
        """Load both trained projections, or only the trained visual projection.

        ``image-only`` leaves the original pretrained text projection and logit
        scale untouched; this reproduces blend weights (image=1.0, text=0.0).
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        apply_projection_adapter(self.clip, state, adapter_mode=adapter_mode)
