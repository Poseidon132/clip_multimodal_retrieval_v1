# VSCode Quick Start

1. Open this folder in VSCode.
2. Install the Python extension.
3. Create and activate `.venv` with Python 3.10 or 3.11.
4. Install CUDA PyTorch first, then `pip install -r requirements.txt`.
5. Run `python scripts/verify_env.py` and confirm the RTX 4060 is visible.
6. Run `python -m src.evaluate --csv data/toy/captions.csv --image-root data/toy/images --split test`.
7. Run `python -m src.features --csv data/toy/captions.csv --image-root data/toy/images --output artifacts/image_text_index.npz`.
8. Run `python -m src.search "a red square" --index artifacts/image_text_index.npz`.
9. Optional UI: `streamlit run app.py`.

First run requires internet access so Transformers can download the CLIP model. The project zip intentionally does not include third-party model weights or Flickr8k images.
