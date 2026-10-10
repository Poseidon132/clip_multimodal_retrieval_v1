from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapter_utils import apply_projection_adapter
from src.index_metadata import get_index_metadata, validate_index_metadata


class ToyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_projection = torch.nn.Linear(3, 2, bias=False)
        self.text_projection = torch.nn.Linear(3, 2, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))


def test_image_only_loads_visual_but_preserves_original_text_and_logit():
    clip = ToyClip()
    original_text = clip.text_projection.weight.detach().clone()
    original_logit = clip.logit_scale.detach().clone()
    trained = ToyClip()
    with torch.no_grad():
        trained.visual_projection.weight.fill_(2)
        trained.text_projection.weight.fill_(3)
        trained.logit_scale.fill_(4)
    state = {
        "visual_projection": trained.visual_projection.state_dict(),
        "text_projection": trained.text_projection.state_dict(),
        "logit_scale": trained.logit_scale.detach().clone(),
    }
    apply_projection_adapter(clip, state, adapter_mode="image-only")
    assert torch.allclose(clip.visual_projection.weight, trained.visual_projection.weight)
    assert torch.allclose(clip.text_projection.weight, original_text)
    assert torch.allclose(clip.logit_scale, original_logit)


def test_both_loads_both_projection_heads_and_logit():
    clip = ToyClip()
    trained = ToyClip()
    with torch.no_grad():
        trained.visual_projection.weight.fill_(2)
        trained.text_projection.weight.fill_(3)
        trained.logit_scale.fill_(4)
    state = {
        "visual_projection": trained.visual_projection.state_dict(),
        "text_projection": trained.text_projection.state_dict(),
        "logit_scale": trained.logit_scale.detach().clone(),
    }
    apply_projection_adapter(clip, state, "both")
    assert torch.allclose(clip.visual_projection.weight, trained.visual_projection.weight)
    assert torch.allclose(clip.text_projection.weight, trained.text_projection.weight)
    assert torch.allclose(clip.logit_scale, trained.logit_scale)


def test_image_only_rejects_full_model():
    with pytest.raises(ValueError, match="projection adapter"):
        apply_projection_adapter(ToyClip(), {"full_state_dict": {}}, "image-only")


def test_index_metadata_prevents_wrong_adapter_mode_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"first")
    metadata = get_index_metadata("model-a", str(checkpoint), "image-only")
    validate_index_metadata(metadata, "model-a", str(checkpoint), "image-only")
    with pytest.raises(ValueError, match="mismatch"):
        validate_index_metadata(metadata, "model-a", str(checkpoint), "both")
    checkpoint.write_bytes(b"second")
    with pytest.raises(ValueError, match="mismatch"):
        validate_index_metadata(metadata, "model-a", str(checkpoint), "image-only")


def test_legacy_index_allowed_only_for_unadapted():
    validate_index_metadata(None, "model-a", None, "both")
    with pytest.raises(ValueError, match="predates"):
        validate_index_metadata(None, "model-a", __file__, "image-only")
    with pytest.raises(ValueError, match="requires"):
        get_index_metadata("model-a", None, "image-only")
