"""Training utilities for safe, split-isolated BLIP image-text matching adaptation.

The CLIP embedding index is used ONLY for selecting training-set hard negatives.
Labels come exclusively from the official training split; validation labels are
used only for model selection. No test image or caption enters optimizer steps.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "clip-blip-itm-adapter-v1"


def normalize_caption(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def word_jaccard(a: str, b: str) -> float:
    aa = set(normalize_caption(a).split())
    bb = set(normalize_caption(b).split())
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / len(aa | bb) if aa | bb else 0.0


def select_trainable_blip_parameters(model: torch.nn.Module, last_text_layers: int = 1) -> list[str]:
    """Freeze vision and most text layers; train ITM head and last N text layers."""
    if last_text_layers < 0:
        raise ValueError("last_text_layers must be >=0")
    for param in model.parameters():
        param.requires_grad_(False)
    if not hasattr(model, "itm_head"):
        raise AttributeError("Expected BlipForImageTextRetrieval.itm_head")
    for param in model.itm_head.parameters():
        param.requires_grad_(True)
    if last_text_layers:
        try:
            layers = model.text_encoder.encoder.layer
        except AttributeError as exc:
            raise AttributeError("Expected BlipForImageTextRetrieval.text_encoder.encoder.layer") from exc
        if last_text_layers > len(layers):
            raise ValueError(f"Requested {last_text_layers} text layers, model has {len(layers)}")
        for layer in layers[-last_text_layers:]:
            for param in layer.parameters():
                param.requires_grad_(True)
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not names:
        raise AssertionError("No BLIP parameters were unfrozen")
    if any(name.startswith("vision_model.") for name in names):
        raise AssertionError("Vision backbone must remain frozen")
    return names


def save_blip_itm_adapter(
    model: torch.nn.Module, path: str | Path, base_model: str,
    config: dict[str, Any], epoch: int, val_mean_r1: float,
) -> None:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError("No trainable parameters to save")
    state = model.state_dict()
    selected = {name: state[name].detach().float().cpu().contiguous() for name in names}
    payload = {
        "format": FORMAT,
        "base_model": str(base_model),
        "trainable_parameter_names": names,
        "state_dict": selected,
        "epoch": int(epoch),
        "validation_mean_R@1": float(val_mean_r1),
        "config": dict(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_blip_itm_adapter(model: torch.nn.Module, path: str | Path, base_model: str) -> dict:
    """Load only saved parameter tensors; refuse shape or base-model mismatch."""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Upgrade PyTorch: secure weights_only=True loading is required") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Unrecognized BLIP ITM adapter checkpoint format")
    if payload.get("base_model") != base_model:
        raise ValueError(f"BLIP checkpoint base model {payload.get('base_model')!r} != {base_model!r}")
    selected = payload.get("state_dict")
    names = payload.get("trainable_parameter_names")
    if not isinstance(selected, dict) or not isinstance(names, list) or set(names) != set(selected):
        raise ValueError("Checkpoint parameter manifest is incomplete")
    expected = model.state_dict()
    for key, value in selected.items():
        if key not in expected:
            raise ValueError(f"Unknown checkpoint parameter: {key}")
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Invalid shape for {key}")
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if unexpected or set(missing) != set(expected).difference(selected):
        raise ValueError("Checkpoint load left unexpected keys")
    return {"epoch": int(payload["epoch"]), "validation_mean_R@1": float(payload["validation_mean_R@1"])}


def _rank_top(scores: torch.Tensor, n: int) -> np.ndarray:
    """Return [B,K] top indices; scores can be on CPU or GPU."""
    return scores.topk(k=min(n, scores.shape[1]), dim=1, largest=True, sorted=True).indices.cpu().numpy()


def _has_lexical_overlap(a: str, b: str, threshold: float) -> bool:
    return normalize_caption(a) == normalize_caption(b) or word_jaccard(a, b) >= threshold


def mine_training_pairs(
    gallery: dict, n_anchor_images: int, seed: int, epoch: int,
    hard_negative_fraction: float = 0.8, candidate_pool: int = 50,
    similar_caption_threshold: float = 0.85, device: str = "cpu",
) -> tuple[list[dict], dict]:
    """Make equal numbers of positive and negative pairs, TRAIN gallery only.

    One positive (a randomly selected of five captions) and one negative per
    anchor. Half of negative examples replace the image, half the caption.
    Hard negatives are mined from frozen CLIP similarities; random negatives
    provide diversity. Exact and near-duplicate captions are excluded.
    """
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0,1]")
    if not 0.0 < similar_caption_threshold <= 1.0:
        raise ValueError("similar_caption_threshold must be in (0,1]")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be >= 1")
    names = np.asarray(gallery["image_names"], dtype=str)
    texts = np.asarray(gallery["captions"], dtype=str)
    targets = np.asarray(gallery["targets"], dtype=np.int64)
    positives = gallery["positive_texts"]
    feat_i = np.asarray(gallery["image_features"], dtype=np.float32)
    feat_t = np.asarray(gallery["text_features"], dtype=np.float32)
    if len(names) < 2 or len(texts) < 2:
        raise ValueError("Need >=2 images and captions for negative mining")
    if len(names) != len(feat_i) or len(texts) != len(feat_t):
        raise ValueError("Gallery features do not align with names and captions")
    if len(targets) != len(texts) or len(positives) != len(names):
        raise ValueError("Malformed gallery labels")
    if any(not group for group in positives):
        raise ValueError("Every training image must have at least one caption")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate training image identities")
    if n_anchor_images < 1:
        raise ValueError("n_anchor_images must be >= 1")
    rng = np.random.default_rng(seed + 10007 * epoch)
    selected_images = np.sort(rng.choice(len(names), size=min(n_anchor_images, len(names)), replace=False))
    pos_text_ids = np.array([int(rng.choice(positives[int(i)])) for i in selected_images], dtype=np.int64)

    torch_device = torch.device(device)
    with torch.no_grad():
        # Gallery feature matrices are ~75MB in this dataset and no gradients are needed.
        all_i = torch.as_tensor(feat_i, device=torch_device)
        all_t = torch.as_tensor(feat_t, device=torch_device)
        ranking_caption, ranking_image = [], []
        for start in range(0, len(selected_images), 64):
            idx_i = selected_images[start:start + 64]
            idx_t = pos_text_ids[start:start + 64]
            ranking_caption.extend(_rank_top(all_i[idx_i] @ all_t.T, candidate_pool).tolist())
            ranking_image.extend(_rank_top(all_t[idx_t] @ all_i.T, candidate_pool).tolist())
        del all_i, all_t

    examples: list[dict] = []
    stats = Counter()
    for j, (img_i, text_i) in enumerate(zip(selected_images, pos_text_ids)):
        img_i, text_i = int(img_i), int(text_i)
        anchor_name = str(names[img_i])
        pos_caption = str(texts[text_i])
        examples.append({"image": anchor_name, "caption": pos_caption,
                         "label": 1, "kind": "positive", "anchor_image": anchor_name})
        neg_type = "image" if (j + epoch) % 2 == 0 else "text"
        try_hard = rng.random() < hard_negative_fraction
        neg = None
        if neg_type == "image":
            was_hard = False
            if try_hard:
                for k in ranking_image[j]:
                    if int(k) != img_i:
                        neg = str(names[int(k)])
                        was_hard = True
                        break
            if neg is None:
                for k in rng.permutation(len(names)):
                    if int(k) != img_i:
                        neg = str(names[int(k)])
                        was_hard = True
                        break
            if neg is None:
                raise RuntimeError("No valid negative image")
            examples.append({"image": neg, "caption": pos_caption, "label": 0,
                             "kind": "hard_image" if was_hard else "random_image",
                             "anchor_image": anchor_name})
        else:
            same_image_captions = [str(texts[k]) for k in positives[img_i]]
            def acceptable(k):
                k = int(k)
                return int(targets[k]) != img_i and not any(
                    _has_lexical_overlap(str(texts[k]), own, similar_caption_threshold)
                    for own in same_image_captions
                )
            if try_hard:
                for k in ranking_caption[j]:
                    if acceptable(k):
                        neg = int(k)
                        break
            was_hard = neg is not None
            if neg is None:
                for k in rng.permutation(len(texts)):
                    if acceptable(k):
                        neg = int(k)
                        break
            if neg is None:
                raise RuntimeError("No valid negative caption")
            examples.append({"image": anchor_name, "caption": str(texts[neg]), "label": 0,
                             "kind": "hard_text" if was_hard else "random_text",
                             "anchor_image": anchor_name})
        stats[examples[-1]["kind"]] += 1
    if len(examples) != 2 * len(selected_images):
        raise AssertionError("Must produce one positive and one negative per anchor")
    if sum(row["label"] for row in examples) * 2 != len(examples):
        raise AssertionError("Positive/negative classes are not balanced")
    details = {"anchor_images": len(selected_images), "pairs": len(examples),
               "positive_pairs": len(selected_images), "negative_pairs": len(selected_images),
               "negative_types": dict(stats), "epoch": epoch, "seed": seed,
               "uses_split": "train"}
    random.Random(seed + epoch).shuffle(examples)
    return examples, details
