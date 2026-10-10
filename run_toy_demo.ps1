$ErrorActionPreference = "Stop"
python scripts/make_toy_dataset.py
python scripts/verify_env.py
python -m src.evaluate --csv data/toy/captions.csv --image-root data/toy/images --split test --output artifacts/toy_zero_shot.json
python -m src.features --csv data/toy/captions.csv --image-root data/toy/images --output artifacts/image_text_index.npz
python -m src.search --text "a red square" --index artifacts/image_text_index.npz --top-k 5
Write-Host "Done. Optional UI: streamlit run app.py"
