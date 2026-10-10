import torch

from src.losses import hard_negative_margin_loss, symmetric_clip_loss


def test_losses_are_finite():
    image_features = torch.eye(4)
    text_features = torch.eye(4)
    logit_scale = torch.tensor(1.0)
    loss = symmetric_clip_loss(image_features, text_features, logit_scale)
    hard = hard_negative_margin_loss(image_features, text_features, margin=0.1)
    assert torch.isfinite(loss)
    assert torch.isfinite(hard)
    assert hard.item() == 0.0
