@echo off
python scripts\make_toy_dataset.py || exit /b 1
python scripts\verify_env.py || exit /b 1
python -m src.evaluate --csv data\toy\captions.csv --image-root data\toy\images --split test --output artifacts\toy_zero_shot.json || exit /b 1
python -m src.features --csv data\toy\captions.csv --image-root data\toy\images --output artifacts\image_text_index.npz || exit /b 1
python -m src.search --text "a red square" --index artifacts\image_text_index.npz --top-k 5 || exit /b 1
echo Done. Optional UI: streamlit run app.py
