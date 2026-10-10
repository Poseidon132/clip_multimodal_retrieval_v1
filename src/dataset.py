from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"image", "caption", "split"}


def load_caption_table(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    df = df.dropna(subset=["image", "caption", "split"]).copy()
    df["image"] = df["image"].astype(str)
    df["caption"] = df["caption"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()
    return df


class GroupedCaptionDataset(Dataset):
    """One image per item, with one caption randomly sampled from its caption set."""

    def __init__(self, csv_path: str | Path, image_root: str | Path, split: str = "train"):
        df = load_caption_table(csv_path)
        df = df[df["split"] == split.lower()].copy()
        if df.empty:
            raise ValueError(f"No rows with split={split!r} in {csv_path}")
        self.image_root = Path(image_root)
        grouped = df.groupby("image")["caption"].apply(list)
        self.items = [(image, captions) for image, captions in grouped.items()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_name, captions = self.items[idx]
        image_path = self.image_root / image_name
        image = Image.open(image_path).convert("RGB")
        caption = random.choice(captions)
        return image, caption, image_name


class CaptionPairDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, split: str = "test"):
        df = load_caption_table(csv_path)
        self.df = df[df["split"] == split.lower()].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows with split={split!r} in {csv_path}")
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.image_root / row["image"]).convert("RGB")
        return image, row["caption"], row["image"]
