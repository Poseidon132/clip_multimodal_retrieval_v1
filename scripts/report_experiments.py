"""Aggregate like-for-like Flickr8k validation/test metrics into CSV and figures.

Use only metrics measured on the same split AND same number of queries.
BLIP sampled-subset exploratory runs cannot be compared with full evaluations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_input_arg(pair: str):
    if "=" not in pair:
        raise ValueError(f"Expected 'MODEL=path.json', got: {pair}")
    name, path = pair.split("=", 1)
    if not name.strip():
        raise ValueError("Model label cannot be blank")
    return name.strip(), Path(path)


def load_metric_result(name: str, path: Path, split: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    declared = data.get("split")
    if declared is not None and declared != split:
        raise ValueError(f"{name}: split={declared!r} does not match requested {split!r}")
    if "selected" in data and "results" in data:
        if "evaluated_text_queries" in data:
            # Reranking: only FULL-split query experiments are comparable.
            if (data["evaluated_text_queries"] != data["gallery_captions"] or
                data["evaluated_image_queries"] != data["gallery_images"]):
                raise ValueError(f"{name}: partial query subset; do not compare it with full-split scores")
        metrics = data["selected"]
    else:
        metrics = data
    keys = [f"{direction}_R@{k}" for direction in ("t2i", "i2t") for k in (1, 5, 10)] + ["mean_R@1"]
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise ValueError(f"{name}: metrics missing {missing}")
    return {
        "model": name,
        **{key: round(float(metrics[key]) * 100, 4) for key in keys},
        "num_images": data.get("num_images"),
        "num_captions": data.get("num_captions"),
        "rerank_latency_t2i_s": data.get("timing", {}).get("blip_t2i_seconds_per_query"),
        "rerank_latency_i2t_s": data.get("timing", {}).get("blip_i2t_seconds_per_query"),
        "peak_gpu_gb": data.get("timing", {}).get("gpu_peak_allocated_gb"),
        "clip_latency_t2i_median_ms": None,
        "clip_latency_i2t_median_ms": None,
        "training_peak_gpu_gb": None,
    }


def write_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "retrieval_comparison.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [entry["model"] for entry in results]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(labels)), 4.5), dpi=150)
    ax.bar(x - width / 2, [r["t2i_R@1"] for r in results], width, label="Text to Image R@1")
    ax.bar(x + width / 2, [r["i2t_R@1"] for r in results], width, label="Image to Text R@1")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_at_1.png", bbox_inches="tight")
    fig.savefig(output_dir / "recall_at_1.svg", bbox_inches="tight")
    plt.close(fig)
    return csv_path


def main():
    p = argparse.ArgumentParser(description="Report comparable CLIP/BLIP retrieval results")
    p.add_argument("--result", action="append", required=True, metavar="MODEL=JSON")
    p.add_argument("--latency", action="append", default=[], metavar="MODEL=JSON",
                   help="Attach benchmark_query_latency.py timing JSON to same-named MODEL")
    p.add_argument("--train-summary", action="append", default=[], metavar="MODEL=JSON",
                   help="Attach train_visual.py run_summary.json to same-named MODEL")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--output-dir", default="artifacts/comparison_val")
    args = p.parse_args()
    results = [load_metric_result(*parse_input_arg(item), args.split) for item in args.result]
    for setting, fields in [
        (args.latency, ("clip_latency_t2i_median_ms", "clip_latency_i2t_median_ms")),
        (args.train_summary, ("training_peak_gpu_gb",)),
    ]:
        for item in setting:
            name, path = parse_input_arg(item)
            matches = [row for row in results if row["model"] == name]
            if len(matches) != 1:
                raise ValueError(f"Timing/summary label {name!r} must match exactly one --result label")
            info = json.loads(path.read_text(encoding="utf-8"))
            if setting is args.latency:
                matches[0][fields[0]] = info["text_to_image"]["median_ms"]
                matches[0][fields[1]] = info["image_to_text"]["median_ms"]
            else:
                matches[0][fields[0]] = info["peak_allocated_gpu_gb"]
    # All rows must have the SAME gallery/caption population when available.
    for key in ("num_images", "num_captions"):
        vals = {result[key] for result in results if result[key] is not None}
        if len(vals) > 1:
            raise ValueError(f"Incomparable {key}: {vals}")
    saved = write_results(results, Path(args.output_dir))
    print(f"Saved {saved} and recall_at_1.png / .svg")
    for r in results:
        print(f"{r['model']:20s} T->I={r['t2i_R@1']:.2f}% I->T={r['i2t_R@1']:.2f}% mean={r['mean_R@1']:.2f}%")


if __name__ == "__main__":
    main()
