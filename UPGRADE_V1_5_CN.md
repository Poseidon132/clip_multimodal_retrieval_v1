# CLIP 多模态检索 v1.5 — BLIP 重排序 / 视觉专项训练 / 实验报告

本补丁在 **v1.4 非对称适配部署版**上安装；不会删除或覆盖 `data/flickr8k`、原来的 `artifacts/flickr8k_best_hn02/adapter.pt` 和任何已有索引。请先备份自己改过的 `app.py`、`src/model.py` 和 `src/search.py`，再把补丁 ZIP 解压到项目根目录，合并文件夹并覆盖同名文件。

已完成：脚本实现、26 个单元/模拟评测测试，以及假 CLIP 两轮训练链路检查。
**未完成**：在你的 Windows + RTX 4060 上运行 BLIP 真实模型、视觉专项三组完整训练；目前不能声称 BLIP 或新训练法已经产生提升。

## 基线（真实结果，禁止和新实验虚构混用）

- Flickr8k 官方划分 Train 6000 / Val 1000 / Test 1000；每张图 5 条 Caption。
- CLIP ViT-B/32 Zero-shot 测试集：T→I R@1 53.78%，I→T R@1 71.10%，Mean R@1 62.44%。
- v1.4 image-only 测试集：T→I R@1 58.54%，I→T R@1 73.10%，Mean R@1 65.82%。
- v1.4 image-only 验证集：T→I R@1 60.28%，I→T R@1 75.40%，Mean R@1 67.84%。

## 安装和测试

在 `E:\AIdemo\clip_multimodal_retrieval_v1` 中解压覆盖，之后运行：

```powershell
python -m pytest -q tests
```

如果没装 pytest：`pip install pytest`。BLIP 仍使用你现有的 `transformers`、`torch` 和 `Pillow`，不需要换掉 CUDA 版 PyTorch。

## 第一阶段：BLIP Top-10 重排序

**1A. 验证集小样本**（从完整 1000 张图 / 5000 条 caption 的 gallery 检索，只抽 100 个文本查询和 20 个图像查询测性能；不会把 gallery 缩小，R@1 是小样本估计值，可能波动）：

```powershell
python scripts/eval_blip_rerank.py `
  --split val `
  --sample-text-queries 100 `
  --sample-image-queries 20 `
  --top-k 10 `
  --batch-size 2 `
  --output artifacts/blip_val_small.json
```

首次需要另外下载约 900 MB 的 `Salesforce/blip-itm-base-coco` 预训练权重；该下载不包含在补丁包里。默认在 CUDA 上用 FP16；如果 OOM，先加 `--batch-size 1`；如果 BLIP 出现 FP16 数值问题，增加 `--precision fp32`。该模型是 **图文匹配 ITM** 模型，不是图片描述生成模型。

`alpha` 是 BLIP 和归一化 CLIP 候选分数的融合权重：0=只按 CLIP，1=只按 BLIP。脚本对 `0, 0.25, 0.5, 0.75, 1.0` 做网格评测，**只推理一次 BLIP**、对同一批候选重复计算多种 alpha。会报告 R@1/5/10、GPU 峰值占用，以及 T→I / I→T 的 **BLIP 增量推理延迟**（不含首次模型加载）。

**注意：** Top-10 重排序只会改变前十名内部名次，不能找回 CLIP 没召回的候选，因此同一批查询的 R@10 必然不变。

**1B. 完整验证集**：小样本确认运行速度、显存、质量可接受之后再运行，候选约 `5000×10 + 1000×10 = 60000` 个图文对，**可能耗时很久**，务必先看 1A 的实际吞吐量估算。命令：

```powershell
python scripts/eval_blip_rerank.py `
  --split val `
  --sample-text-queries 0 `
  --sample-image-queries 0 `
  --batch-size 4 `
  --output artifacts/blip_val_full.json
```

验证集全量输出 `selected.alpha` 才是下一步应固定的最优值。**不要用测试集搜索 alpha**。测试集只能在全部验证实验确认最终配置后执行一次：

```powershell
python scripts/eval_blip_rerank.py `
  --split test `
  --sample-text-queries 0 `
  --sample-image-queries 0 `
  --alpha 0.5 `
  --batch-size 4 `
  --output artifacts/blip_fixed_test.json
```

**将上方 `0.5` 替换为完整验证集选出的 alpha**。不要现在就运行最后一条！

实现说明：直接使用你现有的 `artifacts/flickr8k_asymmetric_index.npz`，自动检查编码器和 `adapter.pt` SHA-256，并通过官方 CSV 过滤出验证集/测试集对应的 gallery，避免意外把训练样本混入评测。BLIP 对 CLIP Top-10 候选做 ITM 分类，用匹配类概率进行融合。

## 第二阶段：冻结文本侧、训练视觉投影、利用 5 条 Caption

本阶段是 **三组单变量消融**，三份配置保持相同学习率、随机种子、Batch 和早停策略：

| 实验 | 每张图 Caption | 预训练特征保持约束 | 训练模块 |
|---|---:|---:|---|
| `visual_single` | 1 | 0 | 只训练视觉投影 |
| `visual_multi` | 5 | 0 | 只训练视觉投影 |
| `visual_multi_reg` | 5 | 0.2 | 只训练视觉投影 |

`visual_multi` 损失：Image→Text 允许同张图的所有 Caption 同时是正样本（logsumexp），Text→Image 则每条文本对应其原图，避免把同一图的不同描述互相当作负样本。

`visual_multi_reg` 另外惩罚微调图像特征偏离原始 CLIP 图像特征；原始预训练投影层作为固定 Teacher，只需 **一次视觉编码**。

训练前可做两批次冒烟检查（输出在 `artifacts/flickr8k_visual_multi_reg/smoke/`，**不得把 smoke 模型当正式结果**）：

```powershell
python scripts/train_visual.py --config configs/flickr8k_visual_multi_reg.yaml --smoke
```

之后按顺序跑三组（每组最多 10 轮，验证集早停 patience 3）：

```powershell
python scripts/train_visual.py --config configs/flickr8k_visual_single.yaml
python scripts/train_visual.py --config configs/flickr8k_visual_multi.yaml
python scripts/train_visual.py --config configs/flickr8k_visual_multi_reg.yaml
```

输出目录各自包含 `adapter.pt`（按验证集 Mean R@1 最佳）、`adapter_last.pt`、`train_history.json`、`best_model.json` 和 `run_summary.json`。用 `--adapter-mode image-only` 加载训练结果。

评测三组**验证集**：

```powershell
python -m src.evaluate --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --split val --adapter artifacts/flickr8k_visual_single/adapter.pt --adapter-mode image-only --output artifacts/visual_single_val.json
python -m src.evaluate --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --split val --adapter artifacts/flickr8k_visual_multi/adapter.pt --adapter-mode image-only --output artifacts/visual_multi_val.json
python -m src.evaluate --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --split val --adapter artifacts/flickr8k_visual_multi_reg/adapter.pt --adapter-mode image-only --output artifacts/visual_multi_reg_val.json
```

原始 v1.4 模型仍保持不变，可随时切回。**仅当新模型验证集更好时**，才考虑给它单独构建索引并接入 Web UI；不要在没重建索引的情况下替换 checkpoint，因为索引有 SHA 校验。

如果 `visual_multi_reg` 在完整验证集胜出，可以为它单独构建索引，再测试这一新版 CLIP 与 BLIP 重排序的组合：

```powershell
python -m src.features --csv data/flickr8k/captions_official.csv --image-root data/flickr8k/Images --adapter artifacts/flickr8k_visual_multi_reg/adapter.pt --adapter-mode image-only --output artifacts/flickr8k_visual_multi_reg_index.npz
python scripts/eval_blip_rerank.py --split val --sample-text-queries 100 --sample-image-queries 20 --index artifacts/flickr8k_visual_multi_reg_index.npz --adapter artifacts/flickr8k_visual_multi_reg/adapter.pt --output artifacts/blip_visual_multi_reg_small.json
```

只有新组合的小样本速度和性能可接受，才决定是否继续跑完整验证集。

## 第三阶段：结果对照、延迟、显存和演示

先测现有 CLIP v1.4 的单查询延迟和推理显存（不含模型加载，使用完整验证 gallery，抽取固定随机查询）：

```powershell
python scripts/benchmark_query_latency.py --split val --output artifacts/clip_val_query_latency.json
```

BLIP 的额外延迟及显存已写进 `blip_val_full.json`。训练显存则写在各 `run_summary.json` 内。注意两者的统计口径不同：CLIP 计单查询编码+索引搜索；BLIP 是在已召回候选上额外进行 Top-10 ITM 推理。

等验证集各项结果都有了后，运行：

```powershell
python scripts/report_experiments.py --split val `
  --result "CLIP=artifacts/flickr8k_val_zero_shot.json" `
  --result "Asymmetric=artifacts/flickr8k_asymmetric_val.json" `
  --result "BLIP-Top10=artifacts/blip_val_full.json" `
  --result "Visual-1=artifacts/visual_single_val.json" `
  --result "Visual-5=artifacts/visual_multi_val.json" `
  --result "Visual-5-Reg=artifacts/visual_multi_reg_val.json" `
  --latency "Asymmetric=artifacts/clip_val_query_latency.json" `
  --train-summary "Visual-5-Reg=artifacts/flickr8k_visual_multi_reg/run_summary.json" `
  --output-dir artifacts/comparison_val
```

输出：`retrieval_comparison.csv`、`recall_at_1.png`、`recall_at_1.svg`。BLIP 小样本文件会被**拒绝**与完整验证集结果拼在一起。后续只有在固定最佳方案后，才在 test split 做统一对照。

**Streamlit**：执行 `streamlit run app.py`，侧边栏可勾选“BLIP Top-10 重排序”，在浏览器进行实测。默认未开启，因为 BLIP 模型下载、显存占用和耗时较大；正式 Web 版本的 `alpha` 要采用完整验证集最终值（手动设置）。

## 版本注意事项

- 本补丁不包括模型权重、Flickr8k 图片、CSV 或旧索引；全部复用你已有数据。
- 项目仍然用 PyTorch + Transformers，支持你现有 RTX 4060 8GB；BLIP 真正推理兼容性需你本地测试。遇到 OOM 优先减小 BLIP batch。
- R@K 是检索召回率，不是分类准确率。样本查询评估没有完整验证集的统计稳定性。
- 新训练实验只以验证集选择参数和最佳 epoch；已经多次看过 test，后续不要为单次 test 指标反复调整模型。
- 模型 `Salesforce/blip-itm-base-coco` 属于独立的第三方预训练权重。第一次运行时请阅读其 Hugging Face model card 与许可（BSD 3-Clause）；不要将模型权重打包进项目仓库。
