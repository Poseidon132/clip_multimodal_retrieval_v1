"""Flickr8k TRAIN-only BLIP ITM fine-tuning; split-isolated val for checkpointing.

Windows / RTX 4060 starting point:
 python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml --check-only
 python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml --smoke
 python scripts/train_blip_itm.py --config configs/flickr8k_blip_itm_last1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blip_itm_training import (
    mine_training_pairs, save_blip_itm_adapter, select_trainable_blip_parameters,
)
from src.blip_rerank import BlipITMScorer
from src.dataset import load_caption_table
from src.search import ExactMultimodalIndex
from src.utils import load_yaml, save_json, set_seed
from scripts.eval_blip_rerank import run as evaluate_reranker, subset_gallery


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], image_root: str | Path):
        self.rows = rows
        self.root = Path(image_root)
        for row in rows:
            if Path(row["image"]).name != row["image"]:
                raise ValueError("Dataset image field must be a plain filename")
            if not (self.root / row["image"]).is_file():
                raise FileNotFoundError(self.root / row["image"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _load_batch(rows, processor, root: Path, device: torch.device, max_length: int):
    images = []
    try:
        for row in rows:
            with Image.open(root / row["image"]) as im:
                images.append(im.convert("RGB"))
        inputs = processor(
            images=images, text=[row["caption"] for row in rows],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
    finally:
        for img in images:
            img.close()
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    return inputs, labels


def _training_mode(model, last_text_layers):
    # Freeze pretrained backbone stochastic behavior; only tuned modules train.
    model.eval()
    model.itm_head.train()
    if last_text_layers:
        for layer in model.text_encoder.encoder.layer[-last_text_layers:]:
            layer.train()


def _train_epoch(
    model, processor, optimizer, scaler, rows, config, device,
    smoke: bool = False,
):
    batch_size = int(config.get("batch_size", 2))
    accumulation = int(config.get("grad_accum_steps", 2))
    if batch_size < 1 or accumulation < 1:
        raise ValueError("batch_size and grad_accum_steps must be >=1")
    ds = PairDataset(rows, config["image_root"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda batch: batch, pin_memory=False)
    _training_mode(model, int(config.get("train_last_text_layers", 1)))
    is_cuda = device.type == "cuda"
    precision = config.get("amp", "auto")
    if precision not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("amp must be auto/bf16/fp16/fp32")
    if not is_cuda and precision in ("bf16", "fp16"):
        raise ValueError("CUDA required for bf16/fp16 training")
    use_bf16 = is_cuda and (precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()))
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 is not supported by this GPU")
    use_fp16 = is_cuda and (precision == "fp16" or (precision == "auto" and not use_bf16))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    enabled_amp = bool(use_bf16 or use_fp16)
    grad_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total, correct, seen, steps = 0.0, 0, 0, 0
    start = time.perf_counter()
    max_steps = 3 if smoke else len(loader)
    for idx, batch_rows in enumerate(tqdm(loader, desc="BLIP ITM train", total=min(max_steps, len(loader)))):
        inputs, targets = _load_batch(batch_rows, processor, ds.root, device,
                                      int(config.get("max_text_length", 35)))
        with torch.autocast(device_type="cuda" if is_cuda else "cpu",
                            dtype=amp_dtype, enabled=enabled_amp):
            output = model(**inputs, use_itm_head=True)
            logits = output.itm_score if hasattr(output, "itm_score") else output[0]
            if logits.ndim != 2 or logits.shape[-1] != 2:
                raise ValueError(f"BLIP ITM logits expected [B,2], got {tuple(logits.shape)}")
            loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite BLIP ITM loss; reduce lr, use bf16, inspect samples")
        scaler.scale(loss / accumulation).backward()
        if (idx + 1) % accumulation == 0 or (idx + 1) == len(loader) or (smoke and idx + 1 == max_steps):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=float(config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        seen += len(batch_rows)
        total += float(loss.item()) * len(batch_rows)
        correct += int((logits.detach().argmax(dim=-1) == targets).sum().item())
        if smoke and idx + 1 >= max_steps:
            break
    return {"train_loss": total / seen, "train_pair_accuracy": correct / seen,
            "pairs_seen": seen, "optimizer_steps": steps,
            "training_seconds": round(time.perf_counter() - start, 2),
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")}


def _eval_args(config: dict, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        csv=config["data_csv"], image_root=config["image_root"],
        index=config["clip_index"], adapter=config["clip_adapter"],
        adapter_mode="image-only", clip_model=config["clip_model"],
        blip_model=config["blip_model"], blip_checkpoint=None,
        split="val", top_k=10, batch_size=int(config.get("val_blip_batch_size", 4)),
        precision="fp32", sample_text_queries=int(config.get("val_sample_text_queries", 250)),
        sample_image_queries=int(config.get("val_sample_image_queries", 80)),
        seed=int(config.get("val_query_seed", 2026)),
        alphas=[1.0], alpha=1.0, output=str(output),
    )


def _validate_split(df):
    split_names = {split: set(df.loc[df["split"] == split, "image"]) for split in ("train", "val", "test")}
    if not all(split_names.values()):
        raise ValueError("Official train, val and test splits are required")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_names[a] & split_names[b]:
            raise ValueError(f"Image leakage between {a} and {b}")
    return {k: len(v) for k, v in split_names.items()}


def train(cfg_path: str, check_only=False, smoke=False):
    cfg = load_yaml(cfg_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    df = load_caption_table(cfg["data_csv"])
    split_counts = _validate_split(df)
    index = ExactMultimodalIndex(cfg["clip_index"])
    index.validate_encoder(cfg["clip_model"], cfg["clip_adapter"], "image-only")
    gallery = subset_gallery(index, cfg["data_csv"], "train")
    if len(gallery["image_names"]) != split_counts["train"]:
        raise ValueError("CLIP index is missing training images")
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    n_images = 24 if smoke else int(cfg.get("anchors_per_epoch", 1200))
    if n_images < 1:
        raise ValueError("anchors_per_epoch must be >=1")
    mine_kwargs = dict(
        gallery=gallery, n_anchor_images=n_images, seed=seed, epoch=1,
        hard_negative_fraction=float(cfg.get("hard_negative_fraction", 0.8)),
        candidate_pool=int(cfg.get("candidate_pool", 50)),
        similar_caption_threshold=float(cfg.get("near_duplicate_jaccard", 0.85)),
        device=str(device),
    )
    pairs, mine_stats = mine_training_pairs(**mine_kwargs)
    print(f"Official splits (images): {split_counts} | used for training ONLY: train")
    print(f"Epoch-1 mining: {mine_stats}")
    if check_only:
        print("CHECK PASSED: No BLIP model loaded and no optimizer step executed.")
        return {"split_counts": split_counts, "mining": mine_stats}

    from transformers import AutoProcessor, BlipForImageTextRetrieval
    print("Loading original BLIP ITM from local Hugging Face cache / Hub...")
    processor = AutoProcessor.from_pretrained(cfg["blip_model"])
    model = BlipForImageTextRetrieval.from_pretrained(cfg["blip_model"]).to(device)
    last_layers = int(cfg.get("train_last_text_layers", 1))
    trained_names = select_trainable_blip_parameters(model, last_layers)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BLIP ITM trainable parameters={param_count:,} ({len(trained_names)} tensors) | last_text_layers={last_layers}")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("learning_rate", 5e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.get("amp", "auto") == "fp16"))
    # auto on a GPU without bf16 requires fp16; enable scaler in that case.
    if device.type == "cuda" and cfg.get("amp", "auto") == "auto" and not torch.cuda.is_bf16_supported():
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    scorer = BlipITMScorer(cfg["blip_model"], device=str(device), precision="fp32",
                           model=model, processor=processor)
    out_dir = Path(cfg.get("output_dir", "artifacts/blip_itm_last1"))
    if smoke:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "train_image_counts": split_counts,
               "checkpoint_selection": "fixed validation-query subset; alpha=1; Top-10; Mean R@1",
               "starting_blip_checkpoint": "Salesforce/blip-itm-base-coco; no previous fine-tune"},
              out_dir / "experiment_manifest.json")
    history = []
    best_score, best_epoch, patience_counter = float("-inf"), 0, 0
    max_epochs = 1 if smoke else int(cfg.get("epochs", 3))
    if max_epochs < 1:
        raise ValueError("epochs must be >=1")
    patience = int(cfg.get("early_stopping_patience", 2))
    if patience < 1:
        raise ValueError("early_stopping_patience must be >=1")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        if epoch > 1:
            pairs, mine_stats = mine_training_pairs(**{**mine_kwargs, "epoch": epoch})
        if cfg.get("save_pair_tables", True) and not smoke:
            pd.DataFrame(pairs).to_csv(out_dir / f"train_pairs_epoch{epoch}.csv", index=False, encoding="utf-8")
        record = {"epoch": epoch, "mining": mine_stats}
        record.update(_train_epoch(model, processor, optimizer, scaler, pairs, cfg, device, smoke=smoke))
        print(f"Epoch {epoch}: loss={record['train_loss']:.4f} | pair_acc={record['train_pair_accuracy']:.3f} "
              f"| {record['pairs_seen']} pairs | precision={record['precision']}")
        if smoke:
            history.append(record)
            save_json(history, out_dir / "train_history.json")
            print("SMOKE TRAINING PASSED. No checkpoint selected and no validation/test used.")
            break
        # Save a last-epoch fallback BEFORE slower validation.
        save_blip_itm_adapter(model, out_dir / "adapter_last.pt", cfg["blip_model"],
                              cfg, epoch=epoch, val_mean_r1=-1.0)
        model.eval()
        val_out = out_dir / f"val_epoch{epoch}.json"
        result = evaluate_reranker(_eval_args(cfg, val_out), scorer=scorer)
        # The evaluator uses an injected model during training, not the published base BLIP.
        # Record its actual provenance so per-epoch JSON is not mistaken for zero-shot BLIP.
        result["blip_checkpoint"] = f"in-memory fine-tuning model after epoch {epoch}"
        save_json(result, val_out)
        val_score = float(result["selected"]["mean_R@1"])
        record["val_mean_R@1"] = val_score
        record["val_t2i_R@1"] = float(result["selected"]["t2i_R@1"])
        record["val_i2t_R@1"] = float(result["selected"]["i2t_R@1"])
        if val_score > best_score + float(cfg.get("min_delta", 0.0001)):
            best_score, best_epoch, patience_counter = val_score, epoch, 0
            save_blip_itm_adapter(model, out_dir / "adapter.pt", cfg["blip_model"],
                                  cfg, epoch=epoch, val_mean_r1=val_score)
            save_json({"best_epoch": epoch, "best_val_subset_mean_R@1": val_score,
                       "eval_text_queries": int(cfg.get("val_sample_text_queries", 250)),
                       "eval_image_queries": int(cfg.get("val_sample_image_queries", 80)),
                       "fixed_alpha": 1.0, "top_k": 10,
                       "blip_checkpoint": str(out_dir / "adapter.pt")},
                      out_dir / "best_model.json")
        else:
            patience_counter += 1
        history.append(record)
        save_json(history, out_dir / "train_history.json")
        print(f"Validation subset Mean R@1={val_score:.4f} | best={best_score:.4f} "
              f"(epoch {best_epoch}) | wait={patience_counter}")
        if patience_counter >= patience:
            print(f"Early stopping after {patience} validation checks without improvement.")
            break
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    save_json({"best_epoch": best_epoch, "best_val_subset_mean_R@1": None if smoke else best_score,
               "epochs_completed": len(history), "trainable_parameters": int(param_count),
               "cuda_peak_allocated_gb": round(peak, 3)}, out_dir / "run_summary.json")
    print(f"Finished | best adapter: {out_dir / 'adapter.pt' if not smoke else '(no adapter in smoke)'} "
          f"| max allocated GPU={peak:.2f} GB")
    return history


def main():
    p = argparse.ArgumentParser(description="Train BLIP ITM reranker using TRAIN-only CLIP negatives")
    p.add_argument("--config", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check-only", action="store_true", help="Verify official split, CLIP index and mined pairs; no BLIP load")
    g.add_argument("--smoke", action="store_true", help="Train only three mini-batches; no val/test or final checkpoint")
    args = p.parse_args()
    train(args.config, check_only=args.check_only, smoke=args.smoke)


if __name__ == "__main__":
    main()
