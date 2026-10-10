"""Compare full validation scores before vs after BLIP ITM fine-tuning.

Refuses incompatible split, gallery, CLIP candidate cutoff, query count and alpha.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("split") != "val":
        raise ValueError(f"Only validation results are allowed: {path}")
    for key in ("selected", "timing", "candidate_top_k"):
        if key not in data:
            raise ValueError(f"Missing {key} in {path}")
    return data


def compare(baseline_path, trained_path):
    before, after = _load(baseline_path), _load(trained_path)
    must_match = (
        "gallery_images", "gallery_captions", "evaluated_text_queries", "evaluated_image_queries",
        "candidate_top_k", "clip_index", "adapter_mode", "query_seed",
    )
    for key in must_match:
        if before.get(key) != after.get(key):
            raise ValueError(f"Cannot compare; incompatible {key}: {before.get(key)!r} vs {after.get(key)!r}")
    if not (before["selected"].get("alpha") == after["selected"].get("alpha") == 1.0):
        raise ValueError("Both models must use predeclared alpha=1.0")
    if before["evaluated_text_queries"] != before["gallery_captions"] or \
       before["evaluated_image_queries"] != before["gallery_images"]:
        raise ValueError("Comparison requires full validation-query evaluation")
    keys = ("t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_R@1")
    print("Metric".ljust(16) + "Original BLIP".rjust(16) + "Fine-tuned".rjust(16) + "Delta(pp)".rjust(15))
    changes = {}
    for key in keys:
        b, a = float(before["selected"][key]), float(after["selected"][key])
        changes[key] = round(100*(a-b), 4)
        print(f"{key:<16}{b*100:>15.2f}%{a*100:>15.2f}%{changes[key]:>+14.2f}")
    print("Decision: use val for model selection; do not use test for hyperparameter search.")
    return changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="artifacts/blip_visual_multi_val_full.json")
    p.add_argument("--finetuned", default="artifacts/flickr8k_blip_itm_last1/val_full.json")
    args = p.parse_args()
    compare(args.baseline, args.finetuned)


if __name__ == "__main__":
    main()
