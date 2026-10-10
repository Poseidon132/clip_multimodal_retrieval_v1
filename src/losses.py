from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100.0) * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hard_negative_margin_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    # Compute hard-negative mining in float32 even when AMP supplies fp16
    # embeddings. This avoids fp16 overflow from very negative masking values.
    sim = image_features.float() @ text_features.float().T
    n = sim.shape[0]
    if n < 2:
        return sim.new_tensor(0.0)

    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    positive = sim.diag()

    # -inf is safe here because sim is float32. The diagonal is excluded from
    # hard-negative selection, while every row still has at least one candidate.
    image_to_text_hard = sim.masked_fill(eye, float("-inf")).max(dim=1).values
    text_to_image_hard = sim.T.masked_fill(eye, float("-inf")).max(dim=1).values

    loss_i = F.relu(margin + image_to_text_hard - positive).mean()
    loss_t = F.relu(margin + text_to_image_hard - positive).mean()
    return 0.5 * (loss_i + loss_t)
