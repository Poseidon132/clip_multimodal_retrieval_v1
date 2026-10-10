import random

import torch
import torch.nn.functional as F
import pytest

from src.losses import symmetric_clip_loss
from src.multi_positive import feature_preservation_loss, multi_positive_clip_loss
from src.visual_data import select_captions


def test_single_caption_loss_equals_symmetric_clip():
    image = F.normalize(torch.randn(5, 8), dim=-1)
    text = F.normalize(torch.randn(5, 8), dim=-1)
    scale = torch.tensor(2.0)
    a = multi_positive_clip_loss(image, text, torch.arange(5), scale)
    b = symmetric_clip_loss(image, text, scale)
    torch.testing.assert_close(a, b)


def test_multi_positive_loss_does_not_penalize_valid_extra_caption():
    image = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    captions = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    labels = torch.tensor([0, 0, 1, 1])
    loss = multi_positive_clip_loss(image, captions, labels, torch.tensor(2.0))
    assert loss.item() < 0.05
    loss.backward()
    assert torch.isfinite(image.grad).all()


def test_multi_positive_amp_fp16_is_finite():
    img = F.normalize(torch.randn(3, 16, dtype=torch.float16), dim=-1)
    txt = F.normalize(torch.randn(6, 16, dtype=torch.float16), dim=-1)
    assert torch.isfinite(multi_positive_clip_loss(img, txt, torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor(1.)))


def test_original_feature_preservation():
    x = F.normalize(torch.randn(5, 4), dim=-1)
    assert feature_preservation_loss(x, x).item() == 0.
    assert feature_preservation_loss(x, -x).item() > 3.99


def test_select_captions_parents_and_seed():
    groups = [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']]
    texts, labels = select_captions(groups, 2, generator=random.Random(3))
    assert len(texts) == 4
    assert labels.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        select_captions(groups, 4)
