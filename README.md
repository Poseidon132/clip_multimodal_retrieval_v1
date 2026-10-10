# CLIP Multimodal Image-Text Retrieval

A compact, resume-ready multimodal project for an RTX 4060-class PC. It uses CLIP to align image and text embeddings, supports text-to-image retrieval, bidirectional Recall@K evaluation, lightweight projection-layer fine-tuning, hard-negative margin loss, and a Streamlit demo.

## What is included

- Zero-shot CLIP evaluation
- Text-to-image retrieval
- Image and text embedding extraction
- Exact cosine search (fast enough for Flickr8k)
- Lightweight projection-layer fine-tuning
- In-batch hard-negative loss
- Recall@1 / Recall@5 / Recall@10 for text-to-image and image-to-text
- Flickr8k preparation utility
- Tiny synthetic dataset for smoke testing
- Streamlit demo
- VSCode launch configuration
- Unit tests

## Recommended hardware

- NVIDIA RTX 4060 8 GB: recommended
- 16 GB system RAM or more
- 8-10 GB free disk for the Python environment, model cache, Flickr8k and outputs

The default model is `openai/clip-vit-base-patch32`. The first run downloads model files from Hugging Face. The model cache is not included in this zip.

## 1. VSCode quick start on Windows

Open this project folder in VSCode, then create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build using the command recommended by the PyTorch website for your CUDA setup, then install the remaining packages:

```powershell
pip install -r requirements.txt
```

Check the GPU:

```powershell
python scripts/verify_env.py
```

## 2. Smoke test with the included toy dataset

The toy images and captions are included, so only the CLIP weights need to be downloaded on first use.

```powershell
python -m src.evaluate --csv data/toy/captions.csv --image-root data/toy/images --split test --output artifacts/toy_zero_shot.json
```

Build an image index:

```powershell
python -m src.features --csv data/toy/captions.csv --image-root data/toy/images --output artifacts/image_text_index.npz
```

Search it:

```powershell
python -m src.search --text "a red square" --index artifacts/image_text_index.npz --top-k 5
```

Or run the bundled script:

```powershell
.\run_toy_demo.ps1
```

## 3. Launch the visual demo

```powershell
streamlit run app.py
```

The default UI uses `artifacts/image_text_index.npz` and `data/toy/images`.

## 4. Use Flickr8k

Download Flickr8k images and captions from a legitimate source and arrange them like this:

```text
data/flickr8k/
  Images/
    1000268201_693b08cb0e.jpg
    ...
  captions.txt
```

The preparation script supports common `captions.txt` CSV format (`image,caption`) and the older `Flickr8k.token.txt` tab-separated format.

```powershell
python scripts/prepare_flickr8k.py --captions data/flickr8k/captions.txt --output data/flickr8k/captions.csv
```

The default split is 6000 train / 1000 validation / 1000 test images.

Zero-shot evaluation:

```powershell
python -m src.evaluate --csv data/flickr8k/captions.csv --image-root data/flickr8k/Images --split test --batch-size 32 --output artifacts/flickr8k_zero_shot.json
```

## 5. Lightweight fine-tuning on RTX 4060

Copy `configs/default.yaml` to `configs/flickr8k.yaml` and update:

```yaml
data_csv: data/flickr8k/captions.csv
image_root: data/flickr8k/Images
output_dir: artifacts/flickr8k_projection
epochs: 5
batch_size: 16
learning_rate: 0.00005
mode: projection
hard_negative_weight: 0.2
```

Run:

```powershell
python -m src.train --config configs/flickr8k.yaml
```

The default `projection` mode freezes the CLIP vision and text backbones and trains only the two projection layers plus the logit scale. This is intentionally lightweight for an 8 GB GPU.

Evaluate the adapter:

```powershell
python -m src.evaluate --csv data/flickr8k/captions.csv --image-root data/flickr8k/Images --split test --adapter artifacts/flickr8k_projection/adapter.pt --output artifacts/flickr8k_finetuned.json
```

## 6. Build the final retrieval index

```powershell
python -m src.features --csv data/flickr8k/captions.csv --image-root data/flickr8k/Images --adapter artifacts/flickr8k_projection/adapter.pt --output artifacts/flickr8k_index.npz
```

Search from the command line:

```powershell
python -m src.search --text "a dog running through grass" --index artifacts/flickr8k_index.npz --adapter artifacts/flickr8k_projection/adapter.pt --top-k 5
```

For the Streamlit UI, set the sidebar image root to `data/flickr8k/Images` and index to `artifacts/flickr8k_index.npz`.

## 7. Evaluation metrics

The evaluation script reports:

- Text-to-image Recall@1 / Recall@5 / Recall@10
- Image-to-text Recall@1 / Recall@5 / Recall@10
- Mean R@1

Do not put unverified numbers in a resume. Run the experiment first and copy the actual metrics from the generated JSON files.

## Project structure

```text
clip_multimodal_retrieval_v1/
  app.py
  requirements.txt
  configs/
  data/toy/
  scripts/
  src/
    dataset.py
    model.py
    losses.py
    metrics.py
    train.py
    evaluate.py
    features.py
    search.py
  tests/
  artifacts/
  .vscode/
```

## Resume wording after you have real results

Suggested format:

> Built a CLIP-based multimodal image-text retrieval system that maps images and natural-language queries into a shared embedding space for text-to-image and image-to-text retrieval; implemented lightweight projection fine-tuning and hard-negative learning, and evaluated retrieval quality with Recall@1/5/10 on Flickr8k.

Add your actual measured improvement only after running the baseline and fine-tuned evaluations.
