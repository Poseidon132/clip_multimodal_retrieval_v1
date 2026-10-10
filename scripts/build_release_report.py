"""Rebuild a transparent CLIP/BLIP project report from recorded metrics.

The seed CSV contains *only* metrics observed in the user's real console logs.
Unknown R@5/R@10 values remain blank unless their original JSON artifacts exist.
All scores in the CSV are percentages (not [0,1] fractions).

Usage: python scripts/build_release_report.py
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

METRICS = (
    "t2i_R@1", "t2i_R@5", "t2i_R@10",
    "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_R@1",
)




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_artifact(source: str, root: Path, split: str) -> dict[str, float] | None:
    if not source:
        return None
    source_path, _, key = source.partition("#")
    file_path = root / source_path
    if not file_path.is_file():
        return None
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if data.get("split") not in (None, split):
        raise ValueError(f"Split mismatch in {source_path}: expected {split}")
    if "evaluated_text_queries" in data:
        if (data["evaluated_text_queries"] != data["gallery_captions"]
                or data["evaluated_image_queries"] != data["gallery_images"]):
            raise ValueError(f"Partial evaluation in {source_path}; not a full-split comparison")
    elif data.get("num_images") not in (None, 1000) or data.get("num_captions") not in (None, 5000):
        raise ValueError(f"Wrong evaluation population in {source_path}")
    if key:
        data = data[key]
    elif "selected" in data and "results" in data:
        data = data["selected"]
    return {m: float(data[m]) * 100 for m in METRICS if m in data}


def merge_source_metrics(rows: list[dict[str, str]], root: Path) -> list[str]:
    notes = []
    for row in rows:
        data = read_artifact(row.get("source_artifact", ""), root, row["split"])
        if data is None:
            continue
        for key, v in data.items():
            observed = row.get(key, "").strip()
            if not observed:
                row[key] = f"{v:.4f}"
            elif abs(float(observed) - v) > 0.03:
                notes.append(f"{row['split']} / {row['model']}: {key} seed={observed}% JSON={v:.4f}% (seed retained)")
    return notes


def metric(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    return float(raw) if raw else None


def fpercent(row: dict[str, str], key: str) -> str:
    val = metric(row, key)
    return f"{val:.2f}%" if val is not None else "—"


def make_chart(rows: list[dict[str, str]], split: str, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    preferred = (
        ["Zero-shot", "Asymmetric HN", "Visual Single", "Visual Multi", "Visual Multi + Reg", "Visual Multi + BLIP"]
        if split == "val" else
        ["Zero-shot", "Asymmetric HN", "Visual Multi", "Visual Multi + BLIP"]
    )
    selection = {r["model"]: r for r in rows if r["split"] == split}
    chosen = [selection[k] for k in preferred if k in selection]
    if not chosen:
        return
    n = len(chosen)
    y = np.arange(n, dtype=float)
    bar_h = 0.32
    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig, ax = plt.subplots(figsize=(10.2, max(3.6, n * .72 + 1.4)), layout="constrained")
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        for k, (field, color, label, shift) in enumerate([
            ("t2i_R@1", "#2F6585", "Text to image R@1", -bar_h / 2),
            ("i2t_R@1", "#D18B52", "Image to text R@1", bar_h / 2),
        ]):
            vals = [metric(r, field) or 0 for r in chosen]
            ax.barh(y + shift, vals, height=bar_h * .9, color=color, label=label)
            for yy, val in zip(y + shift, vals):
                ax.text(val + 1.2, yy, f"{val:.2f}%", va="center", fontsize=9.8, color="#23364A")
        ax.set_yticks(y, [r["model"] for r in chosen], fontsize=10.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 101)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Recall@1 (%)")
        ax.set_title(f"Flickr8k official {split.upper()} | 1,000 images / 5,000 captions", fontsize=13.2,
                     loc="left", pad=16, color="#23364A", weight="bold")
        ax.grid(axis="x", alpha=0.14)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9.6, bbox_to_anchor=(1.0, 1.045))
        outdir.mkdir(parents=True, exist_ok=True)
        png_path = outdir / f"{split}_r1_comparison.png"
        fig.savefig(png_path, dpi=270, facecolor="white")
        fig.savefig(outdir / f"{split}_r1_comparison.svg", facecolor="white")
        plt.close(fig)
        # Explicitly encode an opaque RGB PNG (no accidental alpha channel).
        from PIL import Image
        with Image.open(png_path) as rendered:
            rgb = rendered.convert("RGB")
            rgb.save(png_path, format="PNG", dpi=(270, 270), optimize=True)


def render_markdown(rows: list[dict[str, str]], discrepancies: list[str]) -> str:
    out = [
        "# CLIP × BLIP 跨模态图文检索系统｜实验成果报告",
        "",
        "**版本：v1.6 成果整理稿。** 指标来自实际控制台日志；运行报告脚本时，存在的本地原始 JSON 可补齐缺失项目。",
        "",
        "## 研究任务与实验设置",
        "",
        "- **数据集：** Flickr8k，使用官方划分：Train 6000、Val 1000、Test 1000 张图，每张 5 条文本描述。",
        "- **基座：** `openai/clip-vit-base-patch32`；只更新 `visual_projection`，文本侧保留预训练参数。",
        "- **训练改进：** 多正样本文本（每图 5 条 Caption），按验证集 Mean R@1 保存最佳模型。",
        "- **精排器：** 预训练 `Salesforce/blip-itm-base-coco`，CLIP Top-10 召回后仅在十个候选内进行 BLIP ITM 重排序；验证集选定 `alpha=1.0`。",
        "- **评价：** 文搜图 T→I 与图搜文 I→T 的 Recall@1/5/10；Mean R@1 为两方向 R@1 的算术平均。",
        "- **候选库：** 官方 Val/Test 的每次完整评估都在 1000 图／5000 Caption 库上进行；Web 索引覆盖 8000 图／40000 Caption，因此 Web 不能直接套用测试集成绩。",
        "",
        "## 正式测试集对比",
        "",
    ]
    def write_table(split: str):
        selection = [r for r in rows if r["split"] == split]
        out.append("| 方法 | T→I R@1 | R@5 | R@10 | I→T R@1 | R@5 | R@10 | Mean R@1 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in selection:
            out.append("| " + r["model"] + " | " + " | ".join(fpercent(r, k) for k in METRICS) + " |")
        out.append("")
    write_table("test")
    out += [
        "![Official test R@1 comparison](figures/test_r1_comparison.png)",
        "",
        "**最优模型（官方测试集）：** T→I R@1 **79.32%**；I→T R@1 **90.70%**；Mean R@1 **85.01%**。",
        "与 CLIP Zero-shot **62.44%** 相比提升 **22.57 个百分点**；与 Visual Multi（无 BLIP）**68.11%** 相比提升 **16.90 个百分点**。",
        "这里的 R@1 是检索命中率，不是分类准确率，也不是单次请求匹配概率。",
        "",
        "## 验证集消融",
        "",
    ]
    write_table("val")
    out += [
        "![Official val R@1 comparison](figures/val_r1_comparison.png)",
        "",
        "验证集选择了 Visual Multi 的最佳视觉投影（Epoch 8），并在完整验证集 6000 次查询上选择 BLIP `alpha=1.0`。",
        "Multi 正则项 `0.2` 并未超过无正则的 Visual Multi；这只能说明当前设置没有改善验证集平均表现。",
        "",
        "## 推理效率（RTX 4060 8 GB，本地日志）",
        "",
        "| 指标 | 实测 | 说明 |",
        "|---|---:|---|",
        "| BLIP 文搜图阶段 | 0.194 秒/查询 | Top-10 重排序的平均增量耗时，不含 CLIP 编码/首次模型加载 |",
        "| BLIP 图搜文阶段 | 0.201 秒/查询 | 同上 |",
        "| BLIP GPU 峰值分配显存 | 0.53 GB | 仅脚本报告的相关阶段统计值，不代表整套系统峰值显存 |",
        "",
        "## 结果可追溯与局限",
        "",
        "- 保留 `reports/observed_metrics.csv` 与 `reports/final_comparison.csv`（后一份为本地 JSON 可补全版）；未给出或未找到原始 JSON 的指标保持空白，不使用推测数。",
        "- 固定的验证集最佳 `alpha=1.0` 仅在测试集运行一次固定配置的 BLIP 重排序。此前开发阶段也多次观察过测试集其他模型结果，因此不应宣称全过程是严格盲测。",
        "- BLIP 为独立的 COCO 预训练模型，未在本项目中重新训练；重排序增益不能全部归因于视觉微调。",
        "- Top-10 内重排序无法提高同一候选集的 R@10，R@5/R@1 可通过调整顺序改善。",
        "- 未提供多随机种子/置信区间与外部数据集测试，现阶段不宣称统计显著或跨数据集泛化。",
        "- 若原始 JSON 与用户终端记录冲突，脚本会打印差异并保留用户终端数据，等待人工确认。",
        "",
        "## 项目技术栈",
        "",
        "Python、PyTorch、Transformers、CLIP ViT-B/32、BLIP ITM、对比学习、多正样本损失、向量余弦检索、Streamlit、NumPy、Pandas、Matplotlib。",
        "",
        "## 结果来源（本地原始文件）",
        "",
    ]
    for r in rows:
        out.append(f"- `{r['split']}` · {r['model']}：`{r.get('source_artifact') or 'terminal output only'}`")
    if discrepancies:
        out += ["", "## 需要人工核对的差异", ""]
        out += ["- " + x for x in discrepancies]
    return "\n".join(out) + "\n"


def render_html(rows: list[dict[str, str]]) -> str:
    table_rows = []
    for r in rows:
        if r["split"] != "test":
            continue
        values = "".join(f"<td>{html.escape(fpercent(r, k))}</td>" for k in ("t2i_R@1", "i2t_R@1", "mean_R@1"))
        table_rows.append(f"<tr><th>{html.escape(r['model'])}</th>{values}</tr>")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP × BLIP 检索成果</title><style>
    :root {{font-family: "Segoe UI", "Microsoft YaHei", sans-serif;color:#19354b;background:#f5f7fa}} body{{margin:0}} main{{max-width:1120px;margin:auto;padding:48px 24px}} .hero{{background:#132f44;color:#fff;border-radius:22px;padding:40px}} .hero p{{color:#bbd7df;line-height:1.7}} .tag{{color:#7cd6bd;font-weight:700;letter-spacing:.08em}} h1{{font-size:38px;margin:12px 0}} h2{{margin-top:46px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}} .metric{{background:#fff;border:1px solid #e2e8ef;padding:24px;border-radius:15px;box-shadow:0 4px 15px #29485d0d}} .metric strong{{font-size:32px;color:#196c77;display:block;margin:6px 0}} .metric span{{color:#567387;font-size:14px}} img{{width:100%;background:#fff;border-radius:16px;border:1px solid #e0e8ed}} table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{padding:13px;border-bottom:1px solid #e6edf2;text-align:left}} td{{font-variant-numeric:tabular-nums}} footer{{margin:32px 0;color:#617687;line-height:1.7}} .note{{padding:18px 24px;background:#e8f0f4;border-left:4px solid #218599;border-radius:8px;line-height:1.7}} @media(max-width:650px){{.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}main{{padding:20px}}}}
    </style></head><body><main><section class="hero"><div class="tag">FINAL TEST · FLICKR8K</div><h1>CLIP × BLIP 多模态检索</h1><p>视觉侧多正样本适配 → CLIP Top-10 快速召回 → BLIP 细粒度图文匹配重排序<br>测试库：1000 张图 / 5000 条 Caption。由验证集确定最优配置。</p></section><section class="grid"><div class="metric"><span>文搜图 R@1</span><strong>79.32%</strong><span>测试集</span></div><div class="metric"><span>图搜文 R@1</span><strong>90.70%</strong><span>测试集</span></div><div class="metric"><span>双向 Mean R@1</span><strong>85.01%</strong><span>相对原始 CLIP +22.57 个百分点</span></div></section><h2>正式测试集</h2><div style="overflow:auto"><table><thead><tr><th>方法</th><th>文搜图 R@1</th><th>图搜文 R@1</th><th>Mean R@1</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><p><img src="figures/test_r1_comparison.svg" alt="Flickr8k 测试集各模型双向 Recall@1 对比，最终 BLIP 重排序模型最高"></p><h2>验证集消融</h2><p><img src="figures/val_r1_comparison.svg" alt="验证集各训练策略和 BLIP 重排序的双向 Recall@1 对比"></p><div class="note">以上为检索 Recall@1，不是分类准确率。Web 展示索引为 8000 图 / 40000 Caption，与测试阶段候选库规模不同。BLIP 是额外预训练匹配模型；0.19～0.20 秒为 BLIP Top-10 增量重排序时间。</div><footer>详细数字与来源说明：PROJECT_RESULTS_CN.md · observed_metrics.csv · final_comparison.csv。 未测 R@5/R@10 保留空白；不填补估计值。</footer></main></body></html>'''


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."), help="Project root; default current directory")
    p.add_argument("--output", type=Path, default=Path("reports"), help="Relative to project root")
    args = p.parse_args()
    root = args.root.resolve()
    output = root / args.output
    rows = read_csv(root / "reports" / "observed_metrics.csv")
    if not rows:
        raise ValueError("No rows in observed metrics CSV")
    for row in rows:
        if row["split"] not in ("val", "test"):
            raise ValueError(f"Unexpected split: {row['split']}")
        for key in METRICS:
            value = metric(row, key)
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"Out-of-range metric: {row['model']} {key}={value}")
        t2i, i2t, mean = (metric(row, key) for key in ("t2i_R@1", "i2t_R@1", "mean_R@1"))
        if t2i is not None and i2t is not None and mean is not None and abs((t2i + i2t) / 2 - mean) > 0.025:
            raise ValueError(f"Inconsistent mean_R@1 for {row['split']} / {row['model']}")
    problems = merge_source_metrics(rows, root)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "final_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    figures = output / "figures"
    for split in ("val", "test"):
        make_chart(rows, split, figures)
    (output / "PROJECT_RESULTS_CN.md").write_text(render_markdown(rows, problems), encoding="utf-8")
    (output / "index.html").write_text(render_html(rows), encoding="utf-8")
    manifest = {
        "source_csv": "reports/observed_metrics.csv",
        "optional_raw_sources": [r["source_artifact"] for r in rows if r.get("source_artifact")],
        "generation_script": "scripts/build_release_report.py",
        "metric": "T->I and I->T Recall@1, percentage points",
        "aggregation": "No averaging across runs; each value is a measured single-run score",
        "population": "Flickr8k official Val/Test gallery, 1000 images and 5000 captions per split",
        "plot_transformations": ["select split", "select measured R@1 only", "plot raw percentage values on zero-based bars"],
        "uncertainty": "Not estimated; no repeated seeds or interval measurements provided",
        "missing_values": "Kept blank in CSV and displayed as an em dash in Markdown",
        "alpha": "1.0, selected using full validation only",
        "note": "Pretrained BLIP added as a second-stage component; not an independently trained BLIP model",
    }
    (figures / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output / 'PROJECT_RESULTS_CN.md'}")
    print(f"Created {output / 'index.html'}")
    print(f"Created {output / 'final_comparison.csv'}")
    print(f"Created {figures / 'test_r1_comparison.png'} / .svg")
    print(f"Created {figures / 'val_r1_comparison.png'} / .svg")
    for n in problems:
        print("WARNING: " + n)


if __name__ == "__main__":
    main()
