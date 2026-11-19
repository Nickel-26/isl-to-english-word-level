#!/usr/bin/env python3
"""Evaluate the trained classifier on the official INCLUDE-50 test split.

Produces:
    reports/test_metrics.json        accuracy, macro F1, published comparison
    reports/confusion_matrix.csv     counts, rows = true, cols = predicted
    reports/confusion_matrix.png     normalized heatmap
    reports/per_class_accuracy.csv   recall and support per sign
    reports/most_confused.csv        top off-diagonal confusions

The published 94.5% INCLUDE-50 number comes from a pretrained CNN feature
extractor plus an encoder. This project uses MediaPipe keypoints only, so the
comparison is not like-for-like and is reported with that caveat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import ROOT, add_config_args

from isl.config import ConfigError, load_config
from isl.dataset import KeypointDataset, load_split, synthesize_rest
from isl.model import load_checkpoint_model
from isl.splits import load_splits


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    top = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top[i] for i in range(len(labels))]))


def predict(
    model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    loader = DataLoader(KeypointDataset(features, np.zeros(len(features), dtype=np.int64)),
                        batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            logits = model(batch.to(device))
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default=None, help="Split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--rest-check",
        type=int,
        default=300,
        help="Synthetic rest sequences for a false-positive check (0 disables)",
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        cfg = load_config(args.config, root=args.root)
        report = load_splits(cfg, strict=True)

        split = args.split or str(cfg.get("evaluation.split", "test"))
        checkpoint_path = args.checkpoint or cfg.path("evaluation.checkpoint")
        if not checkpoint_path.is_file():
            raise ConfigError(
                f"Checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
            )

        device = torch.device(
            "cuda"
            if (args.device == "auto" and torch.cuda.is_available())
            else ("cpu" if args.device == "auto" else args.device)
        )
        model, payload = load_checkpoint_model(checkpoint_path, device)
        labels: list[str] = list(payload["labels"])
        sign_hash = str(payload.get("sign_hash", ""))
        if sign_hash and sign_hash != cfg.sign_hash:
            raise ConfigError(
                f"Checkpoint sign_hash {sign_hash} != config {cfg.sign_hash}. "
                "The sign list changed; retrain before evaluating."
            )
        features, y_true, sources = load_split(cfg, split, report)
        probs = predict(model, features, device, args.batch_size)
        y_pred = probs.argmax(axis=1)

        accuracy = float(np.mean(y_pred == y_true))
        top3 = topk_accuracy(probs, y_true, 3)
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
        support = matrix.sum(axis=1)

        reports_dir = cfg.path("evaluation.reports_dir")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Per-class accuracy table (sorted worst first).
        rows = [
            {
                "label": labels[i],
                "display": cfg.display(labels[i]),
                "recall": round(float(per_class[i]), 4),
                "support": int(support[i]),
                "correct": int(matrix[i, i]),
            }
            for i in range(len(labels))
            if support[i] > 0
        ]
        pd.DataFrame(rows).sort_values(["recall", "support"]).to_csv(
            reports_dir / "per_class_accuracy.csv", index=False
        )

        # Most confused off-diagonal pairs.
        pairs = []
        for true_idx in range(len(labels)):
            for pred_idx in range(len(labels)):
                if true_idx == pred_idx or matrix[true_idx, pred_idx] == 0:
                    continue
                pairs.append(
                    {
                        "true": labels[true_idx],
                        "predicted": labels[pred_idx],
                        "count": int(matrix[true_idx, pred_idx]),
                    }
                )
        pairs.sort(key=lambda r: (-r["count"], r["true"], r["predicted"]))
        pd.DataFrame(pairs[:30]).to_csv(
            reports_dir / "most_confused.csv", index=False
        )

        pd.DataFrame(
            matrix, index=labels, columns=labels
        ).to_csv(reports_dir / "confusion_matrix.csv")

        # Heatmap.
        size = max(10, len(labels) * 0.32 + 4)
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            normalized,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[cfg.display(l) for l in labels],
            yticklabels=[cfg.display(l) for l in labels],
            ax=ax,
            cbar_kws={"label": "fraction of true class"},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        ax.set_title(f"INCLUDE-50 {split} confusion matrix (acc {accuracy:.3f})")
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)

        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
        )
        (reports_dir / "classification_report.txt").write_text(
            report_text, encoding="utf-8"
        )

        rest_summary = None
        if args.rest_check > 0 and cfg.rest_enabled:
            rng = np.random.default_rng(int(cfg.get("project.seed", 42)) + 1)
            train_features, _, _ = load_split(cfg, "train", report)
            rest = synthesize_rest(
                train_features, int(args.rest_check), rng=rng
            )
            rest_probs = predict(model, rest, device, args.batch_size)
            rest_idx = labels.index(cfg.rest_label)
            rest_summary = {
                "examples": int(args.rest_check),
                "predicted_rest_rate": float(
                    np.mean(rest_probs.argmax(axis=1) == rest_idx)
                ),
                "mean_max_probability": float(rest_probs.max(axis=1).mean()),
                "note": (
                    "Synthetic rest built from training poses, not real webcam "
                    "no-sign footage. Treat as a sanity check only."
                ),
            }

        published = cfg.get("evaluation.published_accuracy", None)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": payload.get("epoch"),
            "split": split,
            "clips": int(len(y_true)),
            "classes": len(labels),
            "accuracy": round(accuracy, 4),
            "top3_accuracy": round(top3, 4),
            "macro_f1": round(
                float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                4,
            ),
            "published_accuracy": published,
            "published_gap": (
                round(float(published) - accuracy, 4) if published is not None else None
            ),
            "published_caveat": str(cfg.get("evaluation.published_caveat", "")),
            "rest_check": rest_summary,
            "most_confused": pairs[:10],
        }
        (reports_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=1), encoding="utf-8"
        )

        print(f"checkpoint : {checkpoint_path} (epoch {payload.get('epoch')})")
        print(f"split      : {split} ({len(y_true)} clips, {len(labels)} classes)")
        print(f"device     : {device}")
        print()
        print(f"top-1 accuracy : {accuracy:.4f}")
        print(f"top-3 accuracy : {top3:.4f}")
        print(f"macro F1       : {metrics['macro_f1']:.4f}")
        if published is not None:
            print(
                f"published      : {float(published):.4f} "
                f"(gap {metrics['published_gap']:+.4f}, not like-for-like)"
            )
        if rest_summary:
            print(
                f"rest check     : {rest_summary['predicted_rest_rate']:.1%} of "
                f"{rest_summary['examples']} synthetic rest clips predicted as rest"
            )
        print()
        worst = sorted(rows, key=lambda r: (r["recall"], -r["support"]))[:8]
        print("Worst signs:")
        for row in worst:
            print(
                f"  {row['display']:<18} recall {row['recall']:.2f} "
                f"({row['correct']}/{row['support']})"
            )
        print()
        print("Most confused pairs:")
        for pair in pairs[:8]:
            print(
                f"  {cfg.display(pair['true']):<18} -> "
                f"{cfg.display(pair['predicted']):<18} {pair['count']}"
            )
        print()
        print(f"Reports: {reports_dir}")
        print("  test_metrics.json, confusion_matrix.png/.csv,")
        print("  per_class_accuracy.csv, most_confused.csv, classification_report.txt")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
