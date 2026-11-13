#!/usr/bin/env python3
"""Train the keypoint sequence classifier (Stage 3).

Reads the cached keypoints, adds synthetic rest examples, applies
shift/scale/speed augmentation, trains a BiGRU (or the optional Transformer),
and saves the best checkpoint by validation accuracy.

Live progress is written to ``training_progress.txt`` every few batches:

    watch -n 2 cat training_progress.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT, add_config_args

from isl.config import ConfigError, load_config
from isl.dataset import KeypointDataset, build_arrays, class_counts
from isl.model import build_model, parameter_count
from isl.splits import load_splits


def pick_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("--device cuda requested, but CUDA is not available")
    return device


def write_status(
    path: Path,
    *,
    state: str,
    epoch: int,
    total_epochs: int,
    step: int = 0,
    total_steps: int = 0,
    train_loss: float = 0.0,
    train_acc: float = 0.0,
    val_loss: float | None = None,
    val_acc: float | None = None,
    sign_acc: float | None = None,
    rest_acc: float | None = None,
    best_acc: float = 0.0,
    best_epoch: int = 0,
    patience: int = 0,
    lr: float = 0.0,
    elapsed: float = 0.0,
    eta: float | None = None,
    device: torch.device | str = "",
    checkpoint: Path | None = None,
    classes: int = 0,
    params: int = 0,
) -> None:
    lines = [
        "INCLUDE-50 training",
        f"updated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"status       : {state}",
        f"device       : {device}",
        f"classes      : {classes}",
        f"parameters   : {params:,}",
        f"epoch        : {epoch} / {total_epochs}",
    ]
    if total_steps:
        lines.append(
            f"batch        : {step} / {total_steps} "
            f"({100.0 * step / total_steps:.1f}%)"
        )
    lines.extend(
        [
            f"train loss   : {train_loss:.4f}",
            f"train acc    : {train_acc:.4f}",
        ]
    )
    if val_loss is not None:
        lines.append(f"val loss     : {val_loss:.4f}")
    if val_acc is not None:
        lines.append(f"val acc      : {val_acc:.4f}")
    if sign_acc is not None:
        lines.append(f"val sign acc : {sign_acc:.4f}")
    if rest_acc is not None:
        lines.append(f"val rest acc : {rest_acc:.4f}")
    lines.extend(
        [
            f"best score   : {best_acc:.4f} (epoch {best_epoch})",
            f"no-improve   : {patience} epoch(s)",
            f"learning rate: {lr:.2e}",
            f"elapsed      : {int(elapsed)//60:02d}:{int(elapsed)%60:02d}",
        ]
    )
    if eta is not None:
        lines.append(f"eta          : {int(eta)//60:02d}:{int(eta)%60:02d}")
    if checkpoint is not None:
        lines.append(f"checkpoint   : {checkpoint}")
    lines.append("")
    lines.append("Refresh with: watch -n 2 cat training_progress.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rest_idx: int | None,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    sign_correct = 0
    sign_seen = 0
    rest_correct = 0
    rest_seen = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            total_loss += float(criterion(logits, labels)) * len(labels)
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(labels)
            correct += int(matches.sum())
            seen += len(labels)
            if rest_idx is not None:
                is_rest = labels.eq(rest_idx)
                rest_correct += int((matches & is_rest).sum())
                rest_seen += int(is_rest.sum())
                sign_correct += int((matches & ~is_rest).sum())
                sign_seen += int((~is_rest).sum())

    return {
        "loss": total_loss / max(seen, 1),
        "acc": correct / max(seen, 1),
        "sign_acc": (sign_correct / sign_seen) if sign_seen else None,
        "rest_acc": (rest_correct / rest_seen) if rest_seen else None,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    cfg,
    epoch: int,
    metrics: dict,
    best_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "labels": cfg.idx_to_label,
        "sign_hash": cfg.sign_hash,
        "feature_dim": int(cfg.feature_dim),
        "seq_len": int(cfg.get("keypoints.resample_frames", 32)),
        "model_type": str(cfg.get("model.type", "bigru")),
        "model_config": {
            "hidden_size": int(cfg.get("model.hidden_size", 128)),
            "num_layers": int(cfg.get("model.num_layers", 2)),
            "dropout": float(cfg.get("model.dropout", 0.3)),
        },
        "epoch": epoch,
        "metrics": metrics,
        "best_val_acc": best_acc,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Live status file (default: training_progress.txt)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        report = load_splits(cfg, strict=True)
        arrays = build_arrays(cfg, report)

        train_x, train_y = arrays["train"]
        val_x, val_y = arrays["val"]
        test_x, test_y = arrays["test"]

        epochs = int(args.epochs or cfg.get("training.epochs", 80))
        batch_size = int(args.batch_size or cfg.get("training.batch_size", 32))
        lr = float(args.lr or cfg.get("training.lr", 1e-3))
        weight_decay = float(cfg.get("training.weight_decay", 1e-4))
        label_smoothing = float(cfg.get("training.label_smoothing", 0.05))
        patience_limit = int(cfg.get("training.early_stop_patience", 12))
        grad_clip = float(cfg.get("training.grad_clip", 1.0))
        device = pick_device(args.device or str(cfg.get("training.device", "auto")))
        progress_file = args.progress_file or (
            cfg.root / str(cfg.get("training.progress_file", "training_progress.txt"))
        )
        best_path = cfg.path(
            "evaluation.checkpoint", Path("models/checkpoints") / f"{cfg.project_name}_best.pt"
        )
        last_path = best_path.with_name(best_path.stem + "_last" + best_path.suffix)
        history_path = best_path.with_name(best_path.stem + "_history.json")

        seed = int(cfg.get("project.seed", 42))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

        rest_idx = cfg.label_to_idx[cfg.rest_label] if cfg.rest_enabled else None
        num_classes = int(cfg.num_classes)
        counts = class_counts(train_y, num_classes)
        weights = len(train_y) / (num_classes * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=label_smoothing,
        ).to(device)

        model = build_model(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        params = parameter_count(model)

        train_loader = DataLoader(
            KeypointDataset(
                train_x, train_y, cfg=cfg, augment=not args.no_augment, seed=seed
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            KeypointDataset(val_x, val_y), batch_size=batch_size * 2, shuffle=False
        )

        print(cfg.summary())
        print()
        print(f"device          : {device}")
        print(f"model           : {cfg.get('model.type')} ({params:,} parameters)")
        print(f"train           : {train_x.shape}  (incl. synthetic rest)")
        print(f"val             : {val_x.shape}  (incl. synthetic rest)")
        print(f"test (official) : {test_x.shape}")
        print(f"class counts    : min={counts.min()} max={counts.max()}")
        print(f"checkpoint      : {best_path}")
        print(f"progress file   : {progress_file}")
        print()

        started = time.time()
        best_acc = -1.0
        best_epoch = 0
        history: list[dict] = []
        total_batches = len(train_loader)

        write_status(
            progress_file,
            state="starting",
            epoch=0,
            total_epochs=epochs,
            device=device,
            classes=num_classes,
            params=params,
            checkpoint=best_path,
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            seen = 0
            epoch_started = time.time()

            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{epochs}",
                unit="batch",
                leave=False,
            )
            for step, (features, labels) in enumerate(progress, 1):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                batch_size_now = len(labels)
                running_loss += float(loss.detach()) * batch_size_now
                correct += int(logits.argmax(dim=1).eq(labels).sum())
                seen += batch_size_now
                progress.set_postfix(
                    loss=f"{running_loss / max(seen, 1):.3f}",
                    acc=f"{correct / max(seen, 1):.3f}",
                )

                if step % 10 == 0 or step == total_batches:
                    elapsed = time.time() - started
                    done_epochs = (epoch - 1) + step / max(total_batches, 1)
                    eta = elapsed / done_epochs * (epochs - done_epochs) if done_epochs else None
                    write_status(
                        progress_file,
                        state="running",
                        epoch=epoch,
                        total_epochs=epochs,
                        step=step,
                        total_steps=total_batches,
                        train_loss=running_loss / max(seen, 1),
                        train_acc=correct / max(seen, 1),
                        best_acc=best_acc,
                        best_epoch=best_epoch,
                        patience=epoch - 1 - best_epoch,
                        lr=optimizer.param_groups[0]["lr"],
                        elapsed=elapsed,
                        eta=eta,
                        device=device,
                        checkpoint=best_path,
                        classes=num_classes,
                        params=params,
                    )
            progress.close()

            train_loss = running_loss / max(seen, 1)
            train_acc = correct / max(seen, 1)
            val = evaluate(model, val_loader, criterion, device, rest_idx)
            scheduler.step(val["loss"])

            # Balanced accuracy keeps a rest-heavy model from winning: both the
            # 77 real val signs and the synthetic rest rows have to do well.
            if val["sign_acc"] is not None and val["rest_acc"] is not None:
                score = 0.5 * (val["sign_acc"] + val["rest_acc"])
            else:
                score = val["acc"]
            improved = score > best_acc
            if improved:
                best_acc = float(score)
                best_epoch = epoch
                save_checkpoint(
                    best_path,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    metrics=val,
                    best_acc=best_acc,
                )
            save_checkpoint(
                last_path,
                model=model,
                cfg=cfg,
                epoch=epoch,
                metrics=val,
                best_acc=best_acc,
            )

            row = {
                "epoch": epoch,
                "score": round(score, 5),
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                **{k: (round(v, 5) if v is not None else None) for k, v in val.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - epoch_started, 2),
            }
            history.append(row)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    {
                        "project": cfg.project_name,
                        "sign_hash": cfg.sign_hash,
                        "device": str(device),
                        "epochs": history,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )

            elapsed = time.time() - started
            remaining = (elapsed / epoch) * (epochs - epoch)
            write_status(
                progress_file,
                state="running",
                epoch=epoch,
                total_epochs=epochs,
                step=total_batches,
                total_steps=total_batches,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val["loss"],
                val_acc=val["acc"],
                sign_acc=val["sign_acc"],
                rest_acc=val["rest_acc"],
                best_acc=best_acc,
                best_epoch=best_epoch,
                patience=epoch - best_epoch,
                lr=optimizer.param_groups[0]["lr"],
                elapsed=elapsed,
                eta=remaining,
                device=device,
                checkpoint=best_path,
                classes=num_classes,
                params=params,
            )

            marker = " *" if improved else ""
            sign_txt = (
                f"{val['sign_acc']:.4f}" if val["sign_acc"] is not None else "n/a"
            )
            rest_txt = (
                f"{val['rest_acc']:.4f}" if val["rest_acc"] is not None else "n/a"
            )
            print(
                f"epoch {epoch:3d}/{epochs} | train {train_loss:.4f} "
                f"acc {train_acc:.4f} | val {val['loss']:.4f} "
                f"acc {val['acc']:.4f} (sign {sign_txt}, "
                f"rest {rest_txt}){marker}"
            )

            if epoch - best_epoch >= patience_limit:
                print(
                    f"\nEarly stopping: no improvement for {patience_limit} epochs "
                    f"(best epoch {best_epoch}, val acc {best_acc:.4f})."
                )
                break

        elapsed = time.time() - started
        write_status(
            progress_file,
            state="complete",
            epoch=best_epoch,
            total_epochs=epochs,
            step=total_batches,
            total_steps=total_batches,
            train_loss=history[-1]["train_loss"],
            train_acc=history[-1]["train_acc"],
            val_loss=history[-1]["loss"],
            val_acc=history[-1]["acc"],
            sign_acc=history[-1]["sign_acc"],
            rest_acc=history[-1]["rest_acc"],
            best_acc=best_acc,
            best_epoch=best_epoch,
            patience=0,
            lr=optimizer.param_groups[0]["lr"],
            elapsed=elapsed,
            eta=0.0,
            device=device,
            checkpoint=best_path,
            classes=num_classes,
            params=params,
        )

        print()
        print("=" * 70)
        print(
            f"Best validation score: {best_acc:.4f} at epoch {best_epoch} "
            f"({elapsed/60:.1f} min total)"
        )
        print(f"Checkpoint: {best_path}")
        print(f"History   : {history_path}")
        print(f"Status    : {progress_file}")
        print("\nDone. Next: python scripts/evaluate.py")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
