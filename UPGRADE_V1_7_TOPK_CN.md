# v1.7 — BLIP Top-K 候选扩展实验（只补丁，不覆盖模型）

## 目的

现有结果：Flickr8k 官方测试集上，Visual Multi + BLIP ITM Top-10、alpha=1.0，Mean R@1=85.01%。这一数字是用户本地真实运行的已报告成绩，不是本补丁的新成绩。

本补丁只修改 `scripts/eval_blip_rerank.py`，另外增加 `scripts/check_clip_candidate_ceiling.py`；支持 Top-20/30 的正确评测、无需 BLIP 的候选命中率预检查，不调整模型权重和现有 Web。

## 修复问题

旧版无条件断言“BLIP 重排序前后 R@10 必须相同”。此断言只在重排序候选个数恰好为 10 时成立。Top-20/30 可以把原先位于第 11–20/30 位的正样本提升到前 10，从而改变 R@10。新版只在 `--top-k 10` 时检查这一不变量。

新输出 `candidate_oracle_at_top_k`：

- `t2i_candidate_hit_rate`：查询文本的正确图片是否出现在 CLIP Top-K 中的比率。
- `i2t_candidate_hit_rate`：查询图片对应的五条正确文本，是否至少有一条出现在 CLIP Top-K 中的比率。
- `mean_candidate_hit_rate`：以上两个均值。它是 Top-K 候选内理想重排序的 R@1 上限，**不是实际模型准确率**。

## 安装

在 Windows PowerShell 进入 `E:\AIdemo\clip_multimodal_retrieval_v1`。先备份：

```powershell
Copy-Item .\scripts\eval_blip_rerank.py .\scripts\eval_blip_rerank_v16_backup.py
```

将本 ZIP 下载到 Downloads 后用完整路径解压（勿混淆工作目录）：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\clip_blip_v1_7_topk_validation_patch.zip" -DestinationPath . -Force
python -m pytest -q tests
```

若 ZIP 实际下载位置不同，用 `-LiteralPath` 指向真实文件位置。

## 先做候选上限预检查（无需 BLIP 推理，很快）

```powershell
python scripts/check_clip_candidate_ceiling.py `
  --index artifacts/flickr8k_visual_multi_index.npz `
  --adapter artifacts/flickr8k_visual_multi/adapter.pt `
  --adapter-mode image-only `
  --cutoffs 10 20 30
```

如果候选上限几乎不再提升，增加 Top-K 很可能不划算；应考虑改进 BLIP 精细排序质量。

## 再做 Top-20 小样本冒烟实验

```powershell
python scripts/eval_blip_rerank.py `
  --split val `
  --index artifacts/flickr8k_visual_multi_index.npz `
  --adapter artifacts/flickr8k_visual_multi/adapter.pt `
  --adapter-mode image-only `
  --top-k 20 `
  --alpha 1.0 `
  --sample-text-queries 200 `
  --sample-image-queries 40 `
  --batch-size 4 `
  --output artifacts/blip_val_top20_small.json
```

这只是确认正确运行与速度，不用于最终调参结论。

## 用完整验证集做候选数量对照

当前已有的 Top-10 结果：`artifacts/blip_visual_multi_val_full.json`。依次做 Top-20、Top-30：

```powershell
foreach ($k in @(20,30)) {
  python scripts/eval_blip_rerank.py `
    --split val `
    --index artifacts/flickr8k_visual_multi_index.npz `
    --adapter artifacts/flickr8k_visual_multi/adapter.pt `
    --adapter-mode image-only `
    --top-k $k `
    --alpha 1.0 `
    --sample-text-queries 0 `
    --sample-image-queries 0 `
    --batch-size 4 `
    --output "artifacts/blip_visual_multi_val_top${k}.json"
}
```

预计耗时随 Top-K 增加，大致呈线性增长（仅供排期，实际以运行日志为准）。只在 **验证集** 选择 Top-K，最后固定 K 和 alpha=1.0，才能对测试集做一次结果确认。

## 优化判断

首先比较完整验证集：`mean_R@1`、两个方向的 R@1/R@5/R@10、单查询 BLIP 耗时，以及 `candidate_oracle_at_top_k`。

- 若 Top-20 的候选上限显著提高，而实际 R@1 没有提高：瓶颈主要在 BLIP 精排，应考虑使用训练集中的 CLIP 硬负样本进行 ITM 微调（避免把语义等价的其他图片强行标为负样本）。
- 若 Top-20 的实际 R@1 提升显著且延迟可接受：再考虑 Top-30。
- 若 Top-20 下降：不要仅为了候选数更多而上线；保留已测 85.01% 方案。

注意：Top-10 内重排不会改变 R@10；Top-20/30 内重排会改变 R@10，可能升也可能降。

候选库来自验证/测试 split，不等同于 Web 演示的全部 8000 张图 + 40000 条描述；结果不能直接换算到整个图片库。
