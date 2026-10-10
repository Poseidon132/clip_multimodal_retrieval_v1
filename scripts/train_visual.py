"""Flickr8k vision-only CLIP projection fine-tuning + multi-positive/teacher ablations.

Preserves the pretrained text projection; saves a standard adapter checkpoint
compatible with: python -m src.evaluate --adapter-mode image-only ...

For quick verification: python scripts/train_visual.py --config ... --smoke
A smoke run uses a separate output folder and must NOT be used as a benchmark.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpointing import BestScoreTracker
from src.evaluate import evaluate_model
from src.model import ClipRetriever
from src.visual_data import MultiCaptionImageDataset, collate_multi, select_captions
from src.multi_positive import feature_preservation_loss, multi_positive_clip_loss
from src.utils import get_device, load_yaml, save_json, set_seed




def visual_teacher_student_features(model, teacher_projection, images, device):
    """Compute pooled vision backbone ONCE and project with both heads."""
    pixels = model.processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    pooled = model.clip.vision_model(pixel_values=pixels).pooler_output
    student = F.normalize(model.clip.visual_projection(pooled).float(), dim=-1)
    with torch.no_grad():
        teacher = F.normalize(teacher_projection(pooled.detach()).float(), dim=-1)
    return student, teacher


def train(config_file: str, smoke: bool = False):
    cfg = load_yaml(config_file)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device"))
    print(f"Device={device}" + (f" | GPU={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))

    captions_per_image = int(cfg.get("captions_per_image", 1))
    preserve_weight = float(cfg.get("feature_preservation_weight", 0.0))
    if preserve_weight < 0:
        raise ValueError("feature_preservation_weight must be >= 0")
    dataset = MultiCaptionImageDataset(cfg["data_csv"], cfg["image_root"], split="train")
    loader = DataLoader(
        dataset, batch_size=int(cfg.get("batch_size", 12)), shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)), collate_fn=collate_multi,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Batch size exceeds training dataset size")

    model = ClipRetriever(cfg.get("model_name", "openai/clip-vit-base-patch32"), mode="visual-only").to(device)
    if any(p.requires_grad for p in model.clip.text_projection.parameters()):
        raise AssertionError("Text projection must be frozen in visual-only mode")
    # A fixed teacher is an untouched COPY of the original pretrained projection.
    teacher_projection = copy.deepcopy(model.clip.visual_projection).to(device).eval()
    for p in teacher_projection.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()), lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = Path(cfg.get("output_dir", "artifacts/visual_only")) / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = 1 if smoke else int(cfg.get("epochs", 10))
    tracker = None if smoke else BestScoreTracker(patience=int(cfg.get("early_stopping_patience", 3)))
    history = []
    print(f"caption_count_per_image={captions_per_image}, preservation_weight={preserve_weight}, epochs={max_epochs}")
    print("Only visual_projection is trainable; text branch and logit_scale remain pretrained.")
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # Prevent dropout/batchnorm fluctuations in the frozen pretrained backbones.
        model.clip.vision_model.eval()
        model.clip.text_model.eval()
        totals = {"clip_loss": 0.0, "preservation_loss": 0.0, "total_loss": 0.0}
        steps = 0
        for images, caption_groups, _ in tqdm(loader, desc=f"visual epoch {epoch}"):
            captions, parents = select_captions(caption_groups, captions_per_image)
            parents = parents.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                image_embeddings, teacher_embeddings = visual_teacher_student_features(
                    model, teacher_projection, images, device
                )
                with torch.no_grad():
                    text_embeddings = model.encode_texts(captions, device)
                clip_loss = multi_positive_clip_loss(
                    image_embeddings, text_embeddings, parents, model.logit_scale
                )
                if preserve_weight > 0:
                    preserve_loss = feature_preservation_loss(image_embeddings, teacher_embeddings)
                else:
                    preserve_loss = torch.zeros((), device=device)
                loss = clip_loss + preserve_weight * preserve_loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            totals["clip_loss"] += float(clip_loss.detach().cpu())
            totals["preservation_loss"] += float(preserve_loss.detach().cpu())
            totals["total_loss"] += float(loss.detach().cpu())
            for img in images:
                img.close()
            if smoke and steps >= 2:
                break

        record = {"epoch": epoch, **{key: value / steps for key, value in totals.items()},
                  "epoch_seconds": round(time.perf_counter() - epoch_start, 2)}
        print(f"Epoch {epoch}: contrastive={record['clip_loss']:.4f} preserve={record['preservation_loss']:.4f} total={record['total_loss']:.4f}")
        model.save_adapter(out_dir / "adapter_last.pt")
        if tracker is None:
            model.save_adapter(out_dir / "adapter.pt")
            should_stop = False
        else:
            val = evaluate_model(
                model, cfg["data_csv"], cfg["image_root"],
                str(cfg.get("validation_split", "val")),
                batch_size=int(cfg.get("val_batch_size", 32)), show_progress=False,
            )
            score = float(val["mean_R@1"])
            improved, should_stop = tracker.step(score, epoch)
            record["validation"] = val
            record["improved"] = improved
            if improved:
                model.save_adapter(out_dir / "adapter.pt")
                save_json({"best_epoch": epoch, "best_mean_R@1": score, "validation": val,
                           "config": str(config_file)}, out_dir / "best_model.json")
            print(f"Validation mean_R@1={score:.4f} | best={tracker.best_score:.4f} epoch={tracker.best_epoch} | wait={tracker.bad_epochs}")
        history.append(record)
        save_json(history, out_dir / "train_history.json")
        if should_stop:
            print("Early stopping: validation mean_R@1 did not improve.")
            break

    memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if is_cuda else 0.0
    save_json({"config": cfg, "checkpoint_mode": "image-only", "epochs_completed": len(history),
               "best_epoch": tracker.best_epoch if tracker else 1,
               "best_val_mean_R@1": tracker.best_score if tracker else None,
               "peak_allocated_gpu_gb": round(memory_gb, 3),
               "total_epoch_seconds": round(sum(r["epoch_seconds"] for r in history), 2)}, out_dir / "run_summary.json")
    print(f"Completed | best adapter: {out_dir / 'adapter.pt'} | peak GPU={memory_gb:.2f} GB")
    return history


def main():
    parser = argparse.ArgumentParser(description="Visual-only CLIP fine-tuning with grouped positives")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="Only 2 training batches, no val, separate smoke/ output")
    args = parser.parse_args()
    train(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
