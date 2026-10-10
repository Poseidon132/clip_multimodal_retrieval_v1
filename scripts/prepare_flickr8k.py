from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_captions(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
        rest = f.readlines()
    lines = [first] + [x.rstrip("\n") for x in rest]

    if first.lower().startswith("image,caption"):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["image"], row["caption"]))
        return rows

    for line in lines:
        if not line.strip():
            continue
        if "\t" in line:
            left, caption = line.split("\t", 1)
            image = left.split("#", 1)[0]
            rows.append((image, caption.strip()))
        elif "," in line:
            image, caption = line.split(",", 1)
            rows.append((image.strip(), caption.strip()))
        else:
            raise ValueError("Unsupported caption format. Expected captions.txt CSV or Flickr8k.token.txt format.")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True)
    p.add_argument("--output", default="data/flickr8k/captions.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=6000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    args = p.parse_args()

    pairs = read_captions(Path(args.captions))
    images = sorted({x[0] for x in pairs})
    random.Random(args.seed).shuffle(images)
    if args.train + args.val + args.test > len(images):
        raise ValueError(f"Requested split has {args.train + args.val + args.test} images but only {len(images)} were found")

    split = {}
    for x in images[: args.train]:
        split[x] = "train"
    for x in images[args.train : args.train + args.val]:
        split[x] = "val"
    for x in images[args.train + args.val : args.train + args.val + args.test]:
        split[x] = "test"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption", "split"])
        for image, caption in pairs:
            if image in split:
                w.writerow([image, caption, split[image]])
    print(f"Saved normalized caption CSV to {out}")


if __name__ == "__main__":
    main()
