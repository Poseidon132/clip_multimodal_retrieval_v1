from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpointing import BestScoreTracker
from .dataset import GroupedCaptionDataset
from .evaluate import evaluate_model
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

    model = ClipRetriever(
        cfg.get("model_name", "openai/clip-vit-base-patch32"),
        cfg.get("mode", "projection"),
    ).to(device)
    params = list(model.trainable_parameters())
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
    if hard_negative_weight < 0:
        raise ValueError("hard_negative_weight must be >= 0")

    # Backward-compatible: configs without validation_split keep old behavior.
    val_split = cfg.get("validation_split")
    selection_metric = cfg.get("selection_metric", "mean_R@1")
    val_batch_size = int(cfg.get("val_batch_size", 32))
    tracker = None
    if val_split:
        tracker = BestScoreTracker(
            patience=int(cfg.get("early_stopping_patience", 3)),
            min_delta=float(cfg.get("min_delta", 0.0)),
        )
        print(
            f"Validation split: {val_split} | select: {selection_metric} | "
            f"patience: {tracker.patience}"
        )

    max_epochs = int(cfg.get("epochs", 3))
    if max_epochs < 1:
        raise ValueError("epochs must be >= 1")
    for epoch in range(1, max_epochs + 1):
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
                    loss = base_loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.detach().cpu())
            total += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        avg = total / max(1, len(loader))
        record = {"epoch": epoch, "loss": avg}
        print(f"Epoch {epoch}: loss={avg:.4f}")

        if tracker is not None:
            # Save the last weights separately; adapter.pt always means best-val.
            model.save_adapter(out_dir / "adapter_last.pt")
            val_metrics = evaluate_model(
                model,
                cfg["data_csv"],
                cfg["image_root"],
                str(val_split),
                batch_size=val_batch_size,
                show_progress=False,
            )
            if selection_metric not in val_metrics:
                raise KeyError(
                    f"Unknown selection_metric={selection_metric!r}; "
                    f"available keys: {sorted(val_metrics)}"
                )
            score = float(val_metrics[selection_metric])
            improved, should_stop = tracker.step(score, epoch)
            record.update({"val": val_metrics, "improved": improved})
            if improved:
                model.save_adapter(out_dir / "adapter.pt")
                save_json(
                    {
                        "best_epoch": tracker.best_epoch,
                        "selection_metric": selection_metric,
                        "best_score": tracker.best_score,
                        "val": val_metrics,
                        "config": str(config_path),
                    },
                    out_dir / "best_model.json",
                )
            print(
                f"Validation {selection_metric}={score:.4f} | "
                f"best={tracker.best_score:.4f} (epoch {tracker.best_epoch}) | "
                f"no_improvement={tracker.bad_epochs}"
            )
        else:
            # Original behavior for toy or configs with no validation split.
            model.save_adapter(out_dir / "adapter.pt")
            should_stop = False

        history.append(record)
        save_json(history, out_dir / "train_history.json")
        if should_stop:
            print(
                f"Early stopping: no validation improvement for "
                f"{tracker.patience} consecutive epochs."
            )
            break

    if tracker is not None:
        print(
            f"Best validation checkpoint: {out_dir / 'adapter.pt'} | "
            f"epoch={tracker.best_epoch}, {selection_metric}={tracker.best_score:.4f}"
        )
        print(f"Last epoch checkpoint: {out_dir / 'adapter_last.pt'}")
    else:
        print(f"Saved adapter to {out_dir / 'adapter.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
