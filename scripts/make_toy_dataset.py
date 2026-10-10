from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "toy"
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)

objects = [
    ("red_square.png", "red", "square"),
    ("blue_square.png", "blue", "square"),
    ("green_square.png", "green", "square"),
    ("red_circle.png", "red", "circle"),
    ("blue_circle.png", "blue", "circle"),
    ("green_circle.png", "green", "circle"),
    ("red_triangle.png", "red", "triangle"),
    ("blue_triangle.png", "blue", "triangle"),
    ("green_triangle.png", "green", "triangle"),
    ("yellow_square.png", "yellow", "square"),
    ("yellow_circle.png", "yellow", "circle"),
    ("yellow_triangle.png", "yellow", "triangle"),
]

rows = []
for i, (name, color, shape) in enumerate(objects):
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    if shape == "square":
        d.rectangle((64, 64, 192, 192), fill=color)
    elif shape == "circle":
        d.ellipse((64, 64, 192, 192), fill=color)
    else:
        d.polygon([(128, 48), (48, 208), (208, 208)], fill=color)
    im.save(IMG / name)
    split = "train" if i < 8 else "test"
    captions = [
        f"a {color} {shape} on a white background",
        f"an image of a {color} {shape}",
        f"a simple {shape} colored {color}",
    ]
    for caption in captions:
        rows.append((name, caption, split))

with open(OUT / "captions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image", "caption", "split"])
    w.writerows(rows)

print(f"Created {len(objects)} toy images and {len(rows)} caption pairs in {OUT}")
