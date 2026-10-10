# CLIP × BLIP 多模态跨模态检索系统（v1.6）

> CLIP ViT-B/32 图文向量召回 + 视觉投影多正样本微调 + BLIP ITM Top-10 重排序 + Streamlit Web Demo

**Flickr8k 官方测试集（1000 张图、5000 条 Caption）**：T→I R@1 **79.32%**，I→T R@1 **90.70%**，Mean R@1 **85.01%**。原始 CLIP Mean R@1 **62.44%**，整体增益 **+22.57 个百分点**。BLIP 预训练模型为额外组件，不能将提升全部归因于自行训练的 CLIP Adapter。

## 架构

```text
图片 / 英文文本
    │
    ▼
CLIP Vision / Text Encoder (ViT-B/32)
    │ visual_projection 基于 Flickr8k 每图 5 条 Caption 微调
    ▼
L2 向量归一化 → 余弦相似度 → Top-10 快速召回
    │
    ▼
预训练 BLIP ITM 对十个图文候选做联合匹配
    │ alpha=1.0（验证集选择）
    ▼
双向 Top-K 结果 / 计时 / CSV 导出
```

## 在用户已配置 Windows 环境中直接运行

```powershell
cd E:\AIdemo\clip_multimodal_retrieval_v1
streamlit run app.py
```

默认读取本地：

- 数据集：`data/flickr8k/Images` 与 `data/flickr8k/captions_official.csv`
- 训练权重：`artifacts/flickr8k_visual_multi/adapter.pt`
- 索引：`artifacts/flickr8k_visual_multi_index.npz`
- BLIP 模型：`Salesforce/blip-itm-base-coco`（Hugging Face 缓存）

已安装的环境可以直接运行；若是新机器，先安装 `requirements.txt` 和适合 GPU 的 CUDA PyTorch，下载 Flickr8k 与预训练模型，完成训练、构建索引。

## 从零复现必要命令

官方文本数据应包含 `Flickr8k.token.txt` 与 `Flickr_8k.{train,dev,test}Images.txt`，放在 `data/flickr8k/text/` 下，图片在 `data/flickr8k/Images/`：

```powershell
python scripts/prepare_flickr8k_official.py
python scripts/train_visual.py --config configs/flickr8k_visual_multi.yaml
python -m src.features --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --adapter artifacts/flickr8k_visual_multi/adapter.pt --adapter-mode image-only --output artifacts/flickr8k_visual_multi_index.npz
python -m src.evaluate --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --split val --adapter artifacts/flickr8k_visual_multi/adapter.pt --adapter-mode image-only --output artifacts/visual_multi_val.json
python scripts/eval_blip_rerank.py --split val --index artifacts/flickr8k_visual_multi_index.npz --adapter artifacts/flickr8k_visual_multi/adapter.pt --adapter-mode image-only --sample-text-queries 0 --sample-image-queries 0 --top-k 10 --batch-size 4 --output artifacts/blip_visual_multi_val_full.json
```

**请不要**在已经固定实验结果后通过测试集反复调参。完整测试只在验证集选定 alpha 后运行固定的 `--alpha 1.0`。项目记录了开发过程中对测试集的多次查看，不能宣称严格盲测。

## 整理成果和测试

```powershell
python scripts/build_release_report.py
python -m pytest -q tests
```

- `reports/index.html` 可直接本地打开。
- `reports/PROJECT_RESULTS_CN.md` 含所有真实实验对照、适用范围和局限。
- `reports/RESUME_AND_INTERVIEW_CN.md` 含简历表述和技术问答。
- `reports/figures/` 含 2 张 R@1 对比图，PNG + 文字可编辑 SVG。
- `reports/observed_metrics.csv` 只有日志中确实出现的值；脚本可从用户本地 JSON 补齐尚未提供的 R@5/10。

## 展示成绩的边界

- Recall@1 是检索第一名命中率，不是分类 Accuracy；Mean R@1 为两个方向 R@1 的算术平均。
- 评测库是 1000 图 / 5000 条描述；Web 索引有 8000 图 / 40000 描述，结果不可直接同标。
- BLIP 增量重排序延迟实测约 0.194 秒（文搜图）、0.201 秒（图搜文），不包含首次模型加载与 CLIP 编码。
- 复现实验需要相同权重、随机种子、模型版本和划分；不同 Hugging Face 版本可能带来微小浮点和排序差异。
- BLIP 权重、CLIP 权重和 Flickr8k 图片均不随源代码发布。

运行、安装和结果详情见 `RELEASE_V1_6_CN.md`。
