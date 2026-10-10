"""Grouped Flickr8k image/caption examples and sampling helpers."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import load_caption_table


class MultiCaptionImageDataset(Dataset):
    """One unique image with its complete set of captions per item."""

    def __init__(self, csv_path: str, image_root: str, split: str = "train"):
        table = load_caption_table(csv_path)
        table = table[table["split"] == split].copy()
        if table.empty:
            raise ValueError(f"No {split} examples in {csv_path}")
        self.root = Path(image_root)
        self.items = [(str(name), [str(t) for t in captions]) for name, captions in
                      table.groupby("image", sort=False)["caption"].apply(list).items()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, captions = self.items[idx]
        with Image.open(self.root / name) as img:
            image = img.convert("RGB")
        return image, captions, name


def collate_multi(batch):
    ims, caps, names = zip(*batch)
    return list(ims), list(caps), list(names)


def select_captions(caption_groups: list[list[str]], n_per_image: int, generator=None):
    """Sample without replacement, preserving image grouping in flattened labels."""
    if n_per_image < 1:
        raise ValueError("n_per_image must be >= 1")
    texts, parents = [], []
    # The global Python RNG is seeded in set_seed; generator can be provided for tests.
    import random
    rng = generator if generator is not None else random
    for image_id, captions in enumerate(caption_groups):
        if len(captions) < n_per_image:
            raise ValueError(f"Image {image_id} has {len(captions)} captions; requested {n_per_image}")
        chosen = rng.sample(captions, n_per_image) if n_per_image < len(captions) else captions
        for caption in chosen:
            texts.append(caption)
            parents.append(image_id)
    return texts, torch.tensor(parents, dtype=torch.long)

