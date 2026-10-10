from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import GroupedCaptionDataset
from .losses import hard_negative_margin_loss, symmetric_clip_loss
from .model import ClipRetriever
from .utils import get_device, load_yaml, save_json, set_seed


def collate_fn(batch):
    images, captions, names = zip(*batch)
    return list(images), list(captions), list(names)


def train(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    dataset = GroupedCaptionDataset(cfg["data_csv"], cfg["image_root"], split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = ClipRetriever(cfg.get("model_name", "openai/clip-vit-base-patch32"), cfg.get("mode", "projection")).to(device)
    params = [p for p in model.trainable_parameters()]
    if not params:
        raise RuntimeError("No trainable parameters. Use mode=projection or mode=full.")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    out_dir = Path(cfg.get("output_dir", "artifacts/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hard_negative_weight = float(cfg.get("hard_negative_weight", 0.2))

    for epoch in range(1, int(cfg.get("epochs", 3)) + 1):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for images, captions, _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                image_features, text_features = model.encode_pairs(images, captions, device)
                base_loss = symmetric_clip_loss(image_features, text_features, model.logit_scale)
                if hard_negative_weight > 0.0:
                    hard_loss = hard_negative_margin_loss(
                        image_features,
                        text_features,
                        margin=float(cfg.get("hard_negative_margin", 0.1)),
                    )
                    loss = base_loss + hard_negative_weight * hard_loss
                else:
                    # Do not compute hard-negative loss for the pure projection
                    # baseline. Besides being faster, this avoids unnecessary AMP
                    # interactions when the weight is exactly zero.
                    loss = base_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        avg = total / max(1, len(loader))
        history.append({"epoch": epoch, "loss": avg})
        model.save_adapter(out_dir / "adapter.pt")
        save_json(history, out_dir / "train_history.json")
        print(f"Epoch {epoch}: loss={avg:.4f}")

    print(f"Saved adapter to {out_dir / 'adapter.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
