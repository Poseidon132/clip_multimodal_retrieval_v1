# CLIP retrieval v1.3 — best checkpoint + early stopping

## What changed

- `src/train.py`: optional `validation_split` validation after each epoch; saves best validation weights to `adapter.pt`, latest weights to `adapter_last.pt`; supports `selection_metric`, `early_stopping_patience`, `min_delta`. Legacy configs without `validation_split` keep previous behavior.
- `src/evaluate.py`: exposes `evaluate_model` so train-time validation and standalone CLI use exactly the same bidirectional Recall@K logic without loading CLIP twice.
- `src/checkpointing.py`: validation checkpoint selector, no test split used.
- `tests/test_checkpointing.py`: regression tests for best-score selection and early stopping.
- `configs/flickr8k_best_hn02.yaml`: new training configuration (HN 0.2, max 10 epochs, patience 3, metric `mean_R@1`).

## Installation (Windows PowerShell)

Make a backup of your local code files first. Copy the three files in `src/` and the new config from this update into your existing project folder, preserving the relative paths. Do not delete your Flickr8k data, custom official-split preparation script, or existing adapters.

```powershell
cd E:\AIdemo\clip_multimodal_retrieval_v1
python -m src.train --config configs/flickr8k_best_hn02.yaml
```

The new run writes only under `artifacts/flickr8k_best_hn02`, leaving prior experiments intact.

## Outputs

- `artifacts/flickr8k_best_hn02/adapter.pt`: best validation checkpoint
- `artifacts/flickr8k_best_hn02/adapter_last.pt`: weights at last epoch
- `artifacts/flickr8k_best_hn02/best_model.json`: best epoch, validation metric and score
- `artifacts/flickr8k_best_hn02/train_history.json`: training and validation metrics for each epoch

Evaluate the chosen checkpoint on the test split **after** selecting it using validation data:

```powershell
python -m src.evaluate `
  --csv data/flickr8k/captions_official.csv `
  --image-root data/flickr8k/Images `
  --split test `
  --batch-size 32 `
  --adapter artifacts/flickr8k_best_hn02/adapter.pt `
  --output artifacts/flickr8k_best_hn02_test.json
```

## Important note

Best-checkpoint/early-stopping selection is useful for controlling overfitting, but may or may not improve retrieval performance over the original CLIP zero-shot baseline. Record actual validation and test metrics; avoid reporting training loss as retrieval accuracy.
