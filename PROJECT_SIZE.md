# Approximate local size

The source zip itself is very small. Most disk usage comes from third-party dependencies and data downloaded on the user's PC.

Typical local footprint:

- Project source and toy data: less than 10 MB
- CLIP ViT-B/32 cache: roughly 0.6 GB
- Flickr8k images and captions: roughly 1-2 GB depending on package/source
- PyTorch/CUDA Python environment: several GB
- Feature indexes and experiment outputs: normally less than 1 GB for Flickr8k

Recommended free space: 8-10 GB for a comfortable complete setup.
