"""CLIP × BLIP multimodal search demo (v1.6 release dashboard).

Local data + model checkpoints are NOT bundled. The default index and adapter
point to the user's measured Visual Multi configuration (alpha=1.0).
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps

from src.blip_rerank import BlipITMScorer, rerank_top_k
from src.model import ClipRetriever
from src.release_utils import format_score, read_report_rows, rows_to_csv_bytes
from src.search import ExactMultimodalIndex
from src.utils import get_device

APP_VERSION = "1.6"
FINAL_INDEX = "artifacts/flickr8k_visual_multi_index.npz"
FINAL_ADAPTER = "artifacts/flickr8k_visual_multi/adapter.pt"
FINAL_IMAGE_ROOT = "data/flickr8k/Images"
BLIP_MODEL = "Salesforce/blip-itm-base-coco"
FINAL_ALPHA = 1.0  # Selected on the entire official validation split (not test).

st.set_page_config(
    page_title="CLIP × BLIP | Multimodal Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F6F8FB; }
[data-testid="stSidebar"] { background: #EDF3F6; }
.block-container { max-width: 1400px; padding-top: 1.6rem; padding-bottom: 3rem; }
.release-hero {
    background: linear-gradient(110deg, #15384F 0%, #1B5366 70%, #256E71 100%);
    border-radius: 19px; padding: 26px 35px 27px;
    color: #fff; box-shadow: 0 12px 34px rgba(25, 58, 73, .13);
}
.release-eyebrow { color: #A6E2D0; font-size: .83rem; letter-spacing: .10em; font-weight: 700; }
.release-hero h1 { color: #fff; margin: .35rem 0 .45rem; font-size: 2.08rem; font-weight: 740; }
.release-hero p { margin: 0; color: #DAEAED; font-size: .98rem; line-height: 1.7; }
[data-testid="stMetric"] { background: white; border: 1px solid #DFE9EC; border-radius: 14px; padding: 18px 18px 16px; }
[data-testid="stMetricLabel"] { color: #557183; }
[data-testid="stMetricValue"] { color: #196B79; }
.stButton > button[kind="primary"] { border-radius: 10px; background-color: #157780; border-color: #157780; }
.stDownloadButton button { border-radius: 9px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #DDE7EC; }
section[data-testid="stSidebar"] hr { border-color:#CFDCE4; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="加载 CLIP 编码器中…")
def load_model(name: str, adapter: str, mode: str, file_mtime_ns: int):
    # file_mtime_ns intentionally participates in the cache key.
    _ = file_mtime_ns
    device = get_device()
    model = ClipRetriever(name, mode="frozen").to(device).eval()
    if adapter:
        model.load_adapter(adapter, adapter_mode=mode)
    return model, device


@st.cache_resource(show_spinner="读取特征索引中…")
def load_index(path: str, mtime_ns: int):
    _ = mtime_ns
    return ExactMultimodalIndex(path)


@st.cache_resource(show_spinner="加载 BLIP ITM 模型中（首次可能需要下载权重）…")
def load_blip(model_name: str, precision: str):
    return BlipITMScorer(model_name, precision=precision)


def csv_button(key: str, rows: list[dict], name: str) -> None:
    if not rows:
        return
    fields = list(rows[0])
    st.download_button(
        "导出本次检索 CSV",
        data=rows_to_csv_bytes(rows, fields),
        file_name=name,
        mime="text/csv",
        key=key,
        help="带 UTF-8 BOM，支持在 Excel 中打开",
    )


def search_text(query: str, index: ExactMultimodalIndex, model, device,
                root: Path, top_k: int, precise: bool, blip_batch: int, precision: str):
    start = time.perf_counter()
    with torch.inference_mode():
        vector = model.encode_texts([query], device).float().cpu().numpy()[0]
    candidates = index.search_images(vector, max(top_k, 10) if precise else top_k)
    clip_elapsed = time.perf_counter() - start
    blip_elapsed = 0.0
    if precise:
        scorer = load_blip(BLIP_MODEL, precision)
        rerank_started = time.perf_counter()
        originals = [float(s) for _, s in candidates]
        probs = scorer.score_pairs(
            [root / image for image, _ in candidates],
            [query] * len(candidates), batch_size=blip_batch,
        )
        order, weights = rerank_top_k(range(len(candidates)), originals, probs, FINAL_ALPHA)
        candidates = [(candidates[int(i)][0], float(weights[pos])) for pos, i in enumerate(order)]
        blip_elapsed = time.perf_counter() - rerank_started
    rows = [
        {"rank": rank, "image": filename, "score": float(score), "score_type": "BLIP ITM" if precise else "CLIP cosine"}
        for rank, (filename, score) in enumerate(candidates[:top_k], 1)
    ]
    return rows, clip_elapsed, blip_elapsed


def search_image(image: Image.Image, index: ExactMultimodalIndex, model, device,
                 top_k: int, precise: bool, blip_batch: int, precision: str):
    start = time.perf_counter()
    with torch.inference_mode():
        vector = model.encode_images([image], device).float().cpu().numpy()[0]
    candidates = index.search_captions(vector, max(top_k, 10) if precise else top_k)
    clip_elapsed = time.perf_counter() - start
    blip_elapsed = 0.0
    if precise:
        scorer = load_blip(BLIP_MODEL, precision)
        rerank_started = time.perf_counter()
        originals = [float(score) for _, _, score in candidates]
        probs = scorer.score_pairs(
            [image] * len(candidates),
            [caption for caption, _, _ in candidates], batch_size=blip_batch,
        )
        order, weights = rerank_top_k(range(len(candidates)), originals, probs, FINAL_ALPHA)
        candidates = [
            (candidates[int(i)][0], candidates[int(i)][1], float(weights[pos]))
            for pos, i in enumerate(order)
        ]
        blip_elapsed = time.perf_counter() - rerank_started
    rows = [
        {"rank": rank, "caption": caption, "source_image": source, "score": float(score),
         "score_type": "BLIP ITM" if precise else "CLIP cosine"}
        for rank, (caption, source, score) in enumerate(candidates[:top_k], 1)
    ]
    return rows, clip_elapsed, blip_elapsed


with st.sidebar:
    st.markdown("### 检索配置")
    mode = st.radio(
        "检索模式", ["精确 · CLIP + BLIP", "快速 · CLIP"],
        index=0, help="精确：用 BLIP 对 CLIP Top-10 候选重排序；快速：仅 CLIP 向量检索",
    )
    precise = mode.startswith("精确")
    st.caption("精确模式：BLIP α=1.00，按完整验证集结果固定，不对当前查询重新调参。")
    top_k = st.slider("显示前 K 条", min_value=1, max_value=10, value=5)
    with st.expander("模型与索引路径", expanded=False):
        model_name = st.text_input("CLIP 预训练模型", "openai/clip-vit-base-patch32")
        index_path = st.text_input("向量索引", FINAL_INDEX)
        adapter_path = st.text_input("视觉投影 Adapter", FINAL_ADAPTER)
        adapter_mode = st.selectbox("Adapter 加载方式", ["image-only", "both"], index=0)
        image_root_path = st.text_input("Flickr8k 图片路径", FINAL_IMAGE_ROOT)
    with st.expander("推理设置", expanded=False):
        blip_batch = st.selectbox("BLIP 批量大小", [1, 2, 4], index=1)
        precision = st.selectbox("BLIP 精度", ["auto", "fp32"], index=0,
                                 help="auto：CUDA 上 FP16；出现数值问题可尝试 fp32")
    st.divider()
    st.caption("正式测试：1000 张图 / 5000 条描述。Web 使用全部 8000 张图 / 40000 条描述，命中难度不同。")

st.markdown("""
<div class="release-hero">
  <div class="release-eyebrow">MULTIMODAL RETRIEVAL · FINAL RELEASE v1.6</div>
  <h1>CLIP × BLIP 图文跨模态检索</h1>
  <p>视觉侧多正样本微调 · CLIP 快速向量召回 · BLIP Top-10 精细重排序<br>
     可切换快速 / 精确模式，查看检索耗时，并导出 Top-K 结果。</p>
</div>
""", unsafe_allow_html=True)
st.write("")
stat_cols = st.columns(3, gap="medium")
stat_cols[0].metric("测试集 · 文搜图 R@1", "79.32%", "+25.54 pp vs Zero-shot")
stat_cols[1].metric("测试集 · 图搜文 R@1", "90.70%", "+19.60 pp vs Zero-shot")
stat_cols[2].metric("测试集 · Mean R@1", "85.01%", "+22.57 pp vs Zero-shot")
st.caption("上述为官方测试集 1000 图 / 5000 描述上的检索 R@1，不是当前 Web 的逐次准确率。BLIP 为额外预训练匹配模型。")

index = None
index_error = None
index_file = Path(index_path)
image_root = Path(image_root_path)
adapter_file = Path(adapter_path) if adapter_path else None
if not index_file.is_file():
    index_error = f"索引不存在：{index_file}。请运行 README 中的 src.features 命令。"
elif adapter_file and not adapter_file.is_file():
    index_error = f"权重不存在：{adapter_file}。请确认 Visual Multi 已完成训练。"
elif not image_root.is_dir():
    index_error = f"图片目录不存在：{image_root}。"
else:
    try:
        index = load_index(str(index_file), index_file.stat().st_mtime_ns)
        index.validate_encoder(model_name, adapter_path or None, adapter_mode)
    except (ValueError, OSError, KeyError) as exc:
        index_error = f"索引与当前模型不匹配：{exc}。切换权重后必须使用对应的索引。"

current_config = (
    model_name, index_path, index_file.stat().st_mtime_ns if index_file.is_file() else 0,
    adapter_path, adapter_file.stat().st_mtime_ns if adapter_file and adapter_file.is_file() else 0,
    adapter_mode, image_root_path, top_k, precise,
)
if st.session_state.get("_active_release_config") != current_config:
    st.session_state.pop("t2i_result", None)
    st.session_state.pop("i2t_result", None)
    st.session_state["_active_release_config"] = current_config

if index_error:
    st.warning(index_error)
else:
    st.success(f"检索索引已就绪 · {len(index.image_names):,} 张图片 · {len(index.captions):,} 条描述 · {'BLIP Top-10' if precise else 'CLIP 快速模式'}")

tab_text, tab_image, tab_report, tab_guide = st.tabs(["文搜图", "图搜文", "实验结果", "使用说明"])

with tab_text:
    st.subheader("用一句话检索图片")
    st.caption("支持英文自然语言描述；Flickr8k 的 Caption 主要为英文。")
    query = st.text_input("检索词", value="a dog running through the grass", key="t2i_input")
    if st.button("搜索图片", type="primary", disabled=(index is None or not query.strip()), key="go_text"):
        try:
            mtime = adapter_file.stat().st_mtime_ns if adapter_file else 0
            with st.spinner("正在检索候选并排序…"):
                model, device = load_model(model_name, adapter_path, adapter_mode, mtime)
                rows, clip_time, blip_time = search_text(
                    query.strip(), index, model, device, image_root, top_k, precise, blip_batch, precision,
                )
            st.session_state["t2i_result"] = {
                "rows": rows, "clip_time": clip_time, "blip_time": blip_time,
                "query": query.strip(), "precise": precise,
            }
        except Exception as exc:
            st.error(f"检索失败：{exc}")
            st.info("如果 BLIP 显存不足，请在左侧调低 Batch 到 1；检查图片路径与权重。")
    result = st.session_state.get("t2i_result")
    if result:
        st.markdown(f"**上次检索：** `{result['query']}`")
        time_cols = st.columns(3)
        time_cols[0].metric("CLIP 查询 + 检索", f"{result['clip_time']*1000:.0f} ms")
        time_cols[1].metric("BLIP 增量重排序", f"{result['blip_time']*1000:.0f} ms")
        time_cols[2].metric("两阶段合计*", f"{(result['clip_time']+result['blip_time'])*1000:.0f} ms")
        st.caption("*不含首次模型加载、磁盘打开图片、Web 渲染；非完整端到端延迟。")
        cols = st.columns(3, gap="medium")
        for j, row in enumerate(result["rows"]):
            with cols[j % 3]:
                with st.container(border=True):
                    image_path = image_root / str(row["image"])
                    if image_path.is_file():
                        with Image.open(image_path) as source:
                            st.image(source.convert("RGB"), use_container_width=True)
                    else:
                        st.warning("原图不存在")
                    st.markdown(f"**#{row['rank']:02d}** · {row['image']}")
                    st.caption(f"{row['score_type']} 分数：{format_score(row['score'], result['precise'])}")
        csv_button("export_t2i", result["rows"], "clip_blip_text_to_image.csv")

with tab_image:
    st.subheader("上传图片检索描述")
    st.caption("上传本地图片，返回描述候选。若图片不在 Flickr8k 中，匹配结果不保证精确。")
    uploaded = st.file_uploader("选择 PNG、JPG 或 WebP", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
    image = None
    if uploaded is not None:
        try:
            with Image.open(uploaded) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            st.image(image, width=360, caption="查询图片预览")
        except (OSError, ValueError) as exc:
            st.error(f"图片格式无法读取：{exc}")
    current_upload_sig = (uploaded.name, uploaded.size) if uploaded is not None else None
    if st.session_state.get("_image_upload_sig") != current_upload_sig:
        st.session_state.pop("i2t_result", None)
        st.session_state["_image_upload_sig"] = current_upload_sig
    if st.button("搜索文本描述", type="primary", disabled=(index is None or image is None), key="go_image"):
        try:
            mtime = adapter_file.stat().st_mtime_ns if adapter_file else 0
            with st.spinner("正在检索候选并排序…"):
                model, device = load_model(model_name, adapter_path, adapter_mode, mtime)
                rows, clip_time, blip_time = search_image(
                    image, index, model, device, top_k, precise, blip_batch, precision,
                )
            st.session_state["i2t_result"] = {"rows": rows, "clip_time": clip_time,
                                               "blip_time": blip_time, "precise": precise}
        except Exception as exc:
            st.error(f"检索失败：{exc}")
            st.info("出现显存不足可减小 BLIP Batch。")
    result = st.session_state.get("i2t_result")
    if result:
        time_cols = st.columns(3)
        time_cols[0].metric("CLIP 查询 + 检索", f"{result['clip_time']*1000:.0f} ms")
        time_cols[1].metric("BLIP 增量重排序", f"{result['blip_time']*1000:.0f} ms")
        time_cols[2].metric("两阶段合计*", f"{(result['clip_time']+result['blip_time'])*1000:.0f} ms")
        st.caption("*不含首次模型加载和 Web 渲染。BLIP ITM 分数非校准的准确率。")
        for row in result["rows"]:
            with st.container(border=True):
                col_img, col_text = st.columns([1, 4], gap="medium")
                with col_img:
                    sample_path = image_root / str(row["source_image"])
                    if sample_path.is_file():
                        with Image.open(sample_path) as source:
                            st.image(source.convert("RGB"), use_container_width=True)
                with col_text:
                    st.markdown(f"**#{row['rank']:02d}**　{row['caption']}")
                    st.caption(f"图片来源：{row['source_image']} · {row['score_type']} 分数：{format_score(row['score'], result['precise'])}")
        csv_button("export_i2t", result["rows"], "clip_blip_image_to_text.csv")

with tab_report:
    st.subheader("真实实验结果对比")
    st.caption("全部指标来自 Flickr8k 官方 1000 图 / 5000 Caption 验证或测试集。空白值代表未在当前日志中提供。")
    source = Path("reports/final_comparison.csv")
    if not source.is_file():
        source = Path("reports/observed_metrics.csv")
    if source.is_file():
        rows = read_report_rows(source)
        selected_split = st.radio("数据划分", ["test", "val"], horizontal=True,
                                  format_func=lambda x: "测试集" if x == "test" else "验证集")
        selected = [r for r in rows if r.get("split") == selected_split]
        fields = ["model", "t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_R@1"]
        st.dataframe([{key: r.get(key) for key in fields} for r in selected],
                     use_container_width=True, hide_index=True,
                     column_config={key: st.column_config.NumberColumn(key, format="%.2f%%") for key in fields[1:]})
        plot = Path(f"reports/figures/{selected_split}_r1_comparison.png")
        if plot.is_file():
            st.image(str(plot), use_container_width=True)
        csv_button("export_report", [{key: r.get(key) for key in fields} for r in selected],
                   f"flickr8k_{selected_split}_comparison.csv")
    else:
        st.info("请先运行 `python scripts/build_release_report.py` 生成实验汇总。")
    st.info("BLIP 的改进来自额外的预训练 ITM 重排序器，不能全部归因于 CLIP 微调。Top-10 内重排序不改变 R@10。")

with tab_guide:
    st.subheader("运行方式与注意事项")
    st.markdown("""
    **最终模型路径**
    - `artifacts/flickr8k_visual_multi/adapter.pt`
    - `artifacts/flickr8k_visual_multi_index.npz`
    - BLIP：`Salesforce/blip-itm-base-coco`（首次单独下载）

    **检索模式**
    - **精确**：CLIP 检索 Top-10，BLIP ITM 根据图文匹配分数重排，验证集选定 `alpha=1.0`。
    - **快速**：只用 CLIP，适合快速浏览或 BLIP 尚未下载的场景。

    **测试集成绩的适用范围**
    - 85.01% 是 Flickr8k 官方测试集上 1000 张图 / 5000 Caption 候选库的 Mean R@1。
    - 当前 Web 用 8000 / 40000 全量索引，搜索准确率无法直接等同于该数值。
    - 显示的 CLIP 余弦相似度或 BLIP ITM 分数并非正确率或绝对置信度。

    **生成成果报告**
    ```powershell
    python scripts/build_release_report.py
    ```
    报告位于 `reports/index.html` 和 `reports/PROJECT_RESULTS_CN.md`，支持导出 CSV、PNG、SVG。
    """)
