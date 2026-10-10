# CLIP v1.4 - asymmetric image-only projection deployment

## Result and method

Validation selected image alpha=1.0, text alpha=0.0, using the existing
`artifacts/flickr8k_best_hn02/adapter.pt` trained on Flickr8k.

- Image embeddings: CLIP vision backbone + **fine-tuned visual_projection**.
- Text embeddings: CLIP text backbone + **original pretrained text_projection**.
- Scores: L2-normalized cosine similarity, unchanged Recall@K evaluator.
- No extra training and no new weights needed. This is **asymmetric projection
  loading**, not a new trained residual network.

Reported by the user on the official Flickr8k test set (1000 images, 5000
captions): T2I R@1=58.54%, I2T R@1=73.10%, mean R@1=65.82%.
Zero-shot baseline on the same split: 53.78%, 71.10%, 62.44% respectively.

## Safe installation in existing Windows project

The patch adds or replaces only:

- `src/model.py` (select visual projection or both)
- `src/adapter_utils.py` (new)
- `src/index_metadata.py` (new: fingerprint/check index configuration)
- `src/evaluate.py` (adds `--adapter-mode`)
- `src/features.py` (adds `--adapter-mode` and index metadata)
- `src/search.py` (adds `--adapter-mode` and checks index compatibility)
- `app.py` (defaults to best asymmetric model and checks index compatibility)
- `tests/test_asymmetric_adapter.py` (new)
- `DEPLOY_ASYMMETRIC_V1_4.md` (this file)

**Do not replace** `src/train.py`, `src/losses.py`, configs, user-created
`prepare_flickr8k_official.py`, datasets or existing model checkpoints.

Backup the previous six files before overlaying the patch. Extract the ZIP
**into the project root**, preserving the `src/` and `tests/` folders.

## 1. Optional sanity check: one-pass asymmetric evaluation

```powershell
python -m src.evaluate `
  --csv data/flickr8k/captions_official.csv `
  --image-root data/flickr8k/Images `
  --split val `
  --adapter artifacts/flickr8k_best_hn02/adapter.pt `
  --adapter-mode image-only `
  --batch-size 32 `
  --output artifacts/flickr8k_asymmetric_val.json
```

Expected: near the previously selected validation score T2I R@1 60.28%,
I2T R@1 75.40%. Very small differences may occur because this CLI computes
similarities on CPU using NumPy float32, whereas the feature-blend script
computed the similarities on the GPU.

## 2. Build a single-pass index of all locally available Flickr8k data

```powershell
python -m src.features `
  --csv data/flickr8k/captions_official.csv `
  --image-root data/flickr8k/Images `
  --adapter artifacts/flickr8k_best_hn02/adapter.pt `
  --adapter-mode image-only `
  --batch-size 32 `
  --output artifacts/flickr8k_asymmetric_index.npz
```

This uses all 8000/40000 image/caption records for the **interactive demo
index**, NOT for a training or held-out test claim. No extra learning occurs.
Index metadata remembers the model name, adapter mode, and SHA256 of the
checkpoint. Search refuses to silently use a different configuration.

## 3. Try both retrieval directions in PowerShell

Text-to-image:

```powershell
python -m src.search `
  --text "a dog running through the grass" `
  --index artifacts/flickr8k_asymmetric_index.npz `
  --adapter artifacts/flickr8k_best_hn02/adapter.pt `
  --adapter-mode image-only `
  --top-k 5
```

Image-to-text:

```powershell
$image = (Get-ChildItem .\data\flickr8k\Images\*.jpg | Select-Object -First 1).FullName
python -m src.search `
  --image $image `
  --index artifacts/flickr8k_asymmetric_index.npz `
  --adapter artifacts/flickr8k_best_hn02/adapter.pt `
  --adapter-mode image-only `
  --top-k 5
```

## 4. Web UI

```powershell
streamlit run app.py
```

The new sidebar defaults to the recommended image-only adapter and final
index. To use a different encoder or index, update **both** the model and
index settings; a mismatch raises an explicit error rather than returning
misleading results.

## Experiments and limitations

The mix weights and best checkpoint were selected using validation data;
then the fixed combination was tested once on the held-out Flickr8k test
set. To assess variability, repeat a chosen experiment with a few seeds.
Avoid overstating one-run performance as statistical significance. The
zero-shot R@1 reported by different scripts can differ by roughly 0.1
percentage points due to numerical precision / tie handling.

To undo this patch, restore the backed-up `src/*` files and `app.py`.
