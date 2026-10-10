"""Multi-positive contrastive objective for grouped image-caption supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_clip_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_to_image: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Use *all* captions per image as positives, rather than false negatives.

    I->T: -log(sum positives exp(logit) / sum all captions exp(logit))
    T->I: ordinary CE, as each caption is associated with exactly one image.

    Reduces to the standard symmetric CLIP loss for one caption per image.
    Calculate softmax in fp32 for AMP numerical stability.
    """
    n_images = image_embeddings.size(0)
    n_captions = text_embeddings.size(0)
    labels = caption_to_image.to(image_embeddings.device, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != n_captions:
        raise ValueError("caption_to_image must contain one image index per caption")
    if n_images < 2 or n_captions < 2:
        raise ValueError("Need at least 2 images and 2 captions per batch")
    if (labels < 0).any() or (labels >= n_images).any():
        raise ValueError("caption_to_image includes an invalid image index")
    positives = torch.arange(n_images, device=labels.device)[:, None] == labels[None, :]
    if not positives.any(dim=1).all():
        raise ValueError("Every image in batch needs >= 1 positive caption")
    logits = logit_scale.float().exp().clamp(max=100.0) * (
        image_embeddings.float() @ text_embeddings.float().T
    )
    positive_logsumexp = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    all_logsumexp = torch.logsumexp(logits, dim=1)
    loss_image = (all_logsumexp - positive_logsumexp).mean()
    loss_text = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_image + loss_text)


def feature_preservation_loss(
    tuned_image_embeddings: torch.Tensor,
    original_image_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance between unit-length embeddings (per image)."""
    if tuned_image_embeddings.shape != original_image_embeddings.shape:
        raise ValueError("Original and tuned embeddings must have same shape")
    return (tuned_image_embeddings.float() - original_image_embeddings.float()).square().sum(dim=-1).mean()
