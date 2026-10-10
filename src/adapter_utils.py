"""Helpers for loading a CLIP adapter into selected projection heads.

In asymmetric image-only mode the pretrained text projection is deliberately
left unchanged. This is equivalent to feature-blend image_alpha=1/text_alpha=0.
"""
from __future__ import annotations

import torch


ADAPTER_MODES = ("both", "image-only")


def apply_projection_adapter(clip, state: dict, adapter_mode: str = "both") -> None:
    if adapter_mode not in ADAPTER_MODES:
        raise ValueError(f"adapter_mode must be one of {ADAPTER_MODES}, got {adapter_mode!r}")

    if "full_state_dict" in state:
        if adapter_mode != "both":
            raise ValueError("image-only requires a projection adapter, not a full-model checkpoint")
        clip.load_state_dict(state["full_state_dict"])
        return

    if "visual_projection" not in state:
        raise KeyError("Adapter does not contain visual_projection weights")
    clip.visual_projection.load_state_dict(state["visual_projection"])
    if adapter_mode == "both":
        clip.text_projection.load_state_dict(state["text_projection"])
        with torch.no_grad():
            clip.logit_scale.copy_(state["logit_scale"].to(clip.logit_scale.device))
