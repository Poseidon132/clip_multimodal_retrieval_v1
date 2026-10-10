# CLIP × BLIP v1.6｜最终成果整理 + Streamlit 展示优化

**发布日期：2026-10-09** · **运行环境：Windows 11 / Anaconda / RTX 4060 8 GB**

v1.6 是**在你现有 v1.5 项目上直接覆盖的轻量补丁**：不改变已有模型训练代码、不覆盖任何 Flickr8k 图片、`captions_official.csv`、`adapter.pt` 或 `.npz` 索引，且无需重新训练。

## 1. 安装

将 `clip_multimodal_retrieval_v1_6_release_patch.zip` **解压到**：

```text
E:\AIdemo\clip_multimodal_retrieval_v1
```

覆盖同名 `app.py`；新增 `.streamlit/config.toml`、`src/release_utils.py`、`scripts/build_release_report.py`、`reports/` 等文件。

在覆盖前可备份：

```powershell
Copy-Item .\app.py .\app_v15_backup.py
```

**不要清空** `artifacts/` 或 `data/flickr8k/`。旧模型及之前生成的实验 JSON 需要保留。

## 2. 启动 Web 展示

```powershell
cd E:\AIdemo\clip_multimodal_retrieval_v1
streamlit run app.py
```

访问 `http://localhost:8501`。默认配置已指向：

```text
Index: artifacts/flickr8k_visual_multi_index.npz
Adapter: artifacts/flickr8k_visual_multi/adapter.pt
Adapter mode: image-only
CLIP: openai/clip-vit-base-patch32
BLIP: Salesforce/blip-itm-base-coco
Rerank: Top-10; alpha=1.0（验证集确定）
```

### 界面新功能

- **精确模式**：CLIP Top-10 + BLIP ITM 重排序，作为最终效果展示。
- **快速模式**：仅 CLIP 向量检索，适合低延迟应用及模型未下载时。
- **以文搜图**：三列图片卡片、文件名、名次、匹配分数、CSV 导出。
- **以图搜文**：查询图片预览、候选文本、来源图片缩略图、CSV 导出。
- **耗时显示**：分别统计 CLIP 查询/检索和 BLIP 额外重排序；不含首次模型加载与 Web 渲染。
- **实验看板**：测试集/验证集 R@1/5/10 表格与 PNG 图、实验结果导出。
- **错误检查**：缺少索引/权重时给出提示；索引 SHA 元数据与 Adapter 必须匹配。

**重要：** 首页的 **85.01%** 来自 Flickr8k 官方测试集 **1000 张图 / 5000 条描述**；实际 Web 索引有 **8000 张图 / 40000 条描述**。首页指标是离线参考，不是 Web 动态结果的准确率。

## 3. 重新生成正式成果报告

```powershell
python scripts/build_release_report.py
```

文件：

```text
reports/
  observed_metrics.csv       # 真实终端日志记录，原始保留，缺失值为空
  final_comparison.csv       # 若原始 artifacts/*.json 存在，会补齐缺失的 R@5/R@10
  PROJECT_RESULTS_CN.md      # 中文实验成果报告
  RESUME_AND_INTERVIEW_CN.md # 简历、面试材料
  index.html                 # 可离线浏览的 HTML 成果页面
  figures/
    test_r1_comparison.png   # 测试集可视化
    test_r1_comparison.svg   # SVG 矢量版，文字可编辑
    val_r1_comparison.png    # 验证集可视化
    val_r1_comparison.svg
```

SVG 使用 DejaVu Sans，文字保留为文本对象。没有针对某家期刊进行最终出版规格声明。

从本地 JSON 补齐：你的 `blip_visual_multi_test_final.json` 包含 BLIP 指标，以及同一候选库上 Visual Multi 的 `clip_baseline`，因此报告脚本会把尚未在聊天日志中给出的 R@5/R@10 自动补上。**如果找不到原始 JSON，表中仍留空。**

打开 HTML 报告：

```powershell
Start-Process .\reports\index.html
```

## 4. 工程验证

```powershell
python -m pytest -q tests
python -m py_compile app.py scripts/build_release_report.py src/release_utils.py
```

**限制：** 本次补丁可以在容器中完成语法、数据校验和单元测试，但容器无 Windows RTX 4060 / 原始 Flickr8k 权重与 BLIP 下载缓存。最终请本地检查一次图文检索及耗时显示。

## 5. 汇报口径（实测，不可随意扩大）

| 指标（官方测试集） | 原始 CLIP | Visual Multi | Visual Multi + BLIP |
|---|---:|---:|---:|
| T→I Recall@1 | 53.78% | 60.32% | **79.32%** |
| I→T Recall@1 | 71.10% | 75.90% | **90.70%** |
| Mean Recall@1 | 62.44% | 68.11% | **85.01%** |

总提升为 **22.57 个百分点**。其中 16.90 个百分点来自增加预训练 BLIP 的 Top-10 重排序阶段相对于 Visual Multi 的增益，不能说全部源于微调算法。BLIP 额外推理时间为文搜图约 **0.194 s/query**、图搜文约 **0.201 s/query**（不含 CLIP 编码和首次加载）。

## 6. 外部资源与致谢

- 预训练 CLIP: `openai/clip-vit-base-patch32`；BLIP ITM: `Salesforce/blip-itm-base-coco`。权重/数据集均不包含在补丁里，按照各自的许可证和模型卡使用。
- 可视化完整性流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
