# CLIP × BLIP v1.8 — Flickr8k BLIP ITM 专项微调

## 目标与已知基线

不更换你已经训练好的 CLIP Visual Multi，不重新生成特征索引；只微调第二阶段的 BLIP Image-Text Matching（ITM）排序器。以官方训练集生成配对样本，验证集选择最佳 epoch，测试集只在方案锁定后使用。

原有完整验证集：`Visual Multi + 原始 BLIP Top-10 (alpha=1.0)` 的 T→I R@1 = **80.04%**、I→T R@1 = **89.20%**、Mean R@1 = **84.62%**；原有正式测试集 Mean R@1 为 **85.01%**。这些数字不属于新微调模型，后者结果尚未实测。

### 新模块和不变部分

- **不变：** `artifacts/flickr8k_visual_multi/adapter.pt`、`artifacts/flickr8k_visual_multi_index.npz`、CLIP ViT-B/32 和 Web v1.6 现有精确模式。
- **训练：** BLIP 原始 `Salesforce/blip-itm-base-coco` 的 ITM 分类头以及可选的最后 N 层跨模态文本编码器（默认 N=1）；视觉编码器冻结。
- **负例：** 官方训练集内，CLIP 高相似错误图片 / 错误 Caption，按 1:1 正负配对；兼容 20% 随机负例。过滤同一图片及高度相似的 Caption，以降低错误负例风险（无法完全消除语义上等价的错误负例）。
- **评估：** 每轮固定抽取验证集 250 条文本查询 / 80 条图片查询进行模型选择，完整 1000 图片 / 5000 文本的验证集结果在训练完成后另行计算。
- **硬件：** RTX 4060 8GB，默认 CUDA BF16 自动混合精度；Batch 2，梯度累积 4；OOM 时降低 Batch 并保持梯度累积。
- **权重格式：** 仅保存已训练的参数而非整个约 900MB BLIP 模型；推理时先读取缓存的原始 BLIP，再加载该适配参数，模型名不匹配直接报错。

## 安装（PowerShell，在项目根目录）

把 `clip_blip_itm_v1_8_patch.zip` 先保存到 Windows Downloads 文件夹，确认路径后解压。**不要直接把示例 ZIP 文件名当作已存在的本地文件。**

```powershell
cd E:\AIdemo\clip_multimodal_retrieval_v1

$zip = Join-Path $env:USERPROFILE 'Downloads\clip_blip_itm_v1_8_patch.zip'
Test-Path $zip
# 只有上一行返回 True 才执行下一步：
Expand-Archive -LiteralPath $zip -DestinationPath . -Force

python -m pytest -q tests
```

会覆盖两个兼容文件：`src/blip_rerank.py` 与 `scripts/eval_blip_rerank.py`。本项目的其他模型、索引、数据、Streamlit 页面不变。

## 第一阶段：检查数据与困难负例（不载入 BLIP）

```powershell
python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml --check-only
```

应看到官方 `train=6000 / val=1000 / test=1000`，以及训练对：`anchor_images=1200, pairs=2400`。负样本仅从 **train** 图文候选里挖掘。

## 第二阶段：仅 3 个 Batch 的 GPU 冒烟测试

```powershell
python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml --smoke
```

会真实加载 BLIP 并做前向/反向训练，但 **不运行验证和测试、不产生可宣称有效的最终权重**。检查是否出现 CUDA OOM、数值非有限、BLIP 结构不兼容问题。

### 8GB 显存不够时

修改 `configs/flickr8k_blip_itm_last1.yaml`：

```yaml
batch_size: 1
grad_accum_steps: 8
```

如仍不足：设置 `train_last_text_layers: 0`（只训练 ITM 分类头），或使用 `configs/flickr8k_blip_itm_head_only.yaml` 做低显存消融。

## 第三阶段：建立相同小样本的原始 BLIP 基线

```powershell
python scripts/eval_blip_rerank.py `
  --split val `
  --index artifacts/flickr8k_visual_multi_index.npz `
  --adapter artifacts/flickr8k_visual_multi/adapter.pt `
  --adapter-mode image-only `
  --sample-text-queries 250 `
  --sample-image-queries 80 `
  --seed 2026 `
  --top-k 10 `
  --alpha 1.0 `
  --batch-size 4 `
  --output artifacts/blip_pretrained_fixedsubset_val.json
```

## 第四阶段：正式训练 BLIP ITM

```powershell
python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml
```

默认每轮训练 1200 个锚图像，每图随机 1 条真实 Caption，与一个负例构成 2400 个配对；共最多 3 个 Epoch，验证集 Mean R@1 选最佳且允许早停。训练速度需要在你的 RTX 4060 上测定，不能直接使用此前 CLIP 训练的每轮耗时预估。

输出目录：

```text
artifacts/flickr8k_blip_itm_last1/
  adapter.pt               # 验证集小样本选出的最佳权重
  adapter_last.pt          # 最后一轮参数（不等于最优）
  train_history.json
  best_model.json
  run_summary.json
  experiment_manifest.json
  val_epoch1.json ...
  train_pairs_epoch1.csv ...
```

### 如何判断是否值得继续

先比较 `blip_pretrained_fixedsubset_val.json` 与训练日志中相同固定验证子集的 `val_mean_R@1`。如果没有改善，可以改为只训练 ITM 头的消融版本，或直接保留原始 BLIP；**不要因为训练 Loss 下降，就假定排序变好。**

## 第五阶段：全量验证（不要直接评估 test）

```powershell
python scripts/eval_blip_rerank.py `
  --split val `
  --index artifacts/flickr8k_visual_multi_index.npz `
  --adapter artifacts/flickr8k_visual_multi/adapter.pt `
  --adapter-mode image-only `
  --blip-checkpoint artifacts/flickr8k_blip_itm_last1/adapter.pt `
  --sample-text-queries 0 `
  --sample-image-queries 0 `
  --top-k 10 `
  --alpha 1.0 `
  --batch-size 4 `
  --output artifacts/flickr8k_blip_itm_last1/val_full.json

python scripts/compare_blip_val.py `
  --baseline artifacts/blip_visual_multi_val_full.json `
  --finetuned artifacts/flickr8k_blip_itm_last1/val_full.json
```

全量验证预计和先前 BLIP Top-10 全量验证耗时同一数量级，取决于你的设备实际运行状态。对比脚本会拒绝不同验证集、不同查询数、不同候选库或 alpha 的无效比较。

**如果验证集整体没有超过 84.62%，或者双向性能明显退化，就不要替换当前 85.01% 测试成绩对应的部署模型。**

## 决策后才运行完整测试集

完成完整验证并锁定所有参数后，才允许正式测试：

```powershell
python scripts/eval_blip_rerank.py `
  --split test `
  --index artifacts/flickr8k_visual_multi_index.npz `
  --adapter artifacts/flickr8k_visual_multi/adapter.pt `
  --adapter-mode image-only `
  --blip-checkpoint artifacts/flickr8k_blip_itm_last1/adapter.pt `
  --sample-text-queries 0 `
  --sample-image-queries 0 `
  --top-k 10 `
  --alpha 1.0 `
  --batch-size 4 `
  --output artifacts/flickr8k_blip_itm_last1/test_final.json
```

原项目测试集曾在开发中被多次查看，所以不能将未来结果描述为严格盲测。训练数据未含测试标签，模型选择仍只依据验证集。

## 备注

本次只是 **ITM 二分类损失 + CLIP 挖掘负例** 的参数高效适配；并非重新实现 BLIP 原论文的完整 ITC+ITM+LM 三任务联合预训练，也不是 LoRA。架构介绍可参考 [Hugging Face BLIP 文档](https://huggingface.co/docs/transformers/en/model_doc/blip) 和 [Salesforce 的 BLIP 方法说明](https://www.salesforce.com/blog/blip-bootstrapping-language-image-pretraining/)。
