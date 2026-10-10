# Test report

Local packaging environment checks completed:

- Python byte-code compilation: passed
- Unit tests for dataset parsing: passed
- Unit tests for Recall@K metrics: passed
- Unit tests for CLIP contrastive / hard-negative losses: passed
- Toy dataset generation: passed

Not executed in the packaging environment:

- Download of `openai/clip-vit-base-patch32`
- CUDA inference or fine-tuning
- Flickr8k training/evaluation
- Streamlit browser interaction

Reason: the packaging environment has no bundled third-party model weights and no NVIDIA GPU. These stages are intended to run on the user's VSCode machine. Use `scripts/verify_env.py` first.
