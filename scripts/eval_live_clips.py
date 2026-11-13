#!/usr/bin/env python3
"""Evaluate the live pipeline on labelled single-sign video takes.

Two modes:

``clip``
    Whole clip -> one 32-frame sequence -> one prediction. This is the same
    representation the model was trained on (oracle segmentation) and gives the
    upper bound for a correctly framed take.

``stream``
    Emulate the live demo: a timed sliding window, inference every
    ``--stride`` seconds, EMA smoothing, threshold + streak gate. Stores the
    full probability vector of every inference so thresholds can be re-swept
    offline without re-running MediaPipe.

Also reports per-frame hand/pose presence and whether the pose wrist was inside
the frame. Hand presence is the dominant live failure mode, so it is measured
separately from accuracy.

Example:
    python scripts/eval_live_clips.py --clips-dir eval_vids
    python scripts/eval_live_clips.py --clips-dir eval_vids --mode clip
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_config_args

from isl.config import ConfigError, load_config
from isl.keypoints import Landmarkers, assemble_features, read_video_frames, resample_sequence
from isl.live import DecisionGate, TimedFrameBuffer, buffer_to_sequence, hand_coverage
from isl.model import load_checkpoint_model


def normalise_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalpha())


def top_k(probs: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    order = np.argsort(probs)[::-1][:k]
    return [
        {"label": labels[i], "p": round(float(probs[i]), 4)} for i in order
    ]


def frame_stats(frames) -> dict:
    n = len(frames)
    pose = sum(f.pose is not None for f in frames)
    any_hand = sum(f.left_hand is not None or f.right_hand is not None for f in frames)
    both = sum(f.left_hand is not None and f.right_hand is not None for f in frames)
    left = sum(f.left_hand is not None for f in frames)
    right = sum(f.right_hand is not None for f in frames)
    wrist_in = 0
    wrist_total = 0
    for f in frames:
        if f.pose is None or len(f.pose) < 17:
            continue
        for idx in (15, 16):
            w = f.pose[idx]
            wrist_total += 1
            if 0.0 <= w.x <= 1.0 and 0.0 <= w.y <= 1.0:
                wrist_in += 1
    return {
        "frames": n,
        "pose_rate": round(pose / n, 4) if n else 0.0,
        "any_hand_rate": round(any_hand / n, 4) if n else 0.0,
        "both_hands_rate": round(both / n, 4) if n else 0.0,
        "left_hand_rate": round(left / n, 4) if n else 0.0,
        "right_hand_rate": round(right / n, 4) if n else 0.0,
        "wrist_in_frame_rate": round(wrist_in / wrist_total, 4) if wrist_total else 0.0,
    }


def classify(model, device, sequence: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def evaluate_clip(model, device, frames, cfg, labels, resample_frames, pose_points, use_z, presence_flags, shoulder_visibility, rest_index):
    per_frame = assemble_features(
        frames,
        pose_points=pose_points,
        use_z=use_z,
        presence_flags=presence_flags,
        shoulder_visibility=shoulder_visibility,
    )
    seq = resample_sequence(per_frame, resample_frames, flags=3 if presence_flags else 0)
    probs = classify(model, device, seq)
    ranked = top_k(probs, labels, 3)
    return {
        "top1": ranked[0]["label"],
        "top1_p": ranked[0]["p"],
        "top3": ranked,
        "margin": round(ranked[0]["p"] - ranked[1]["p"], 4),
        "entropy": round(float(-(probs * np.log(probs + 1e-12)).sum()), 4),
        "rest_p": round(float(probs[rest_index]), 4) if rest_index is not None else None,
    }


def stream_clip(
    model, device, frames, fps, cfg, labels, *,
    window, start_after, stride, alpha, resample_frames, pose_points, use_z,
    presence_flags, shoulder_visibility,
):
    """Run the demo inference path; return every inference with full probs.

    Only the model inference is run here. The decision gate is threshold- and
    coverage-independent in its raw input, so re-simulating it from the stored
    probability vectors is both exact and fast.
    """
    buffer = TimedFrameBuffer(window)
    inferences = []
    last_inference = -1e9
    min_frames = 8
    # A short clip can be shorter than start_after; adapt so short takes still
    # produce predictions instead of silently producing zero inferences.
    duration = (len(frames) - 1) / fps if fps > 0 else 0.0
    effective_start = min(start_after, max(0.5 * duration, 0.35))
    for i, frame in enumerate(frames):
        t = i / fps
        buffer.append(t, frame)
        if (
            len(buffer) >= min_frames
            and buffer.duration >= effective_start
            and t - last_inference >= stride
        ):
            last_inference = t
            seq = buffer_to_sequence(
                buffer.frames(),
                resample_frames=resample_frames,
                pose_points=pose_points,
                use_z=use_z,
                presence_flags=presence_flags,
                shoulder_visibility=shoulder_visibility,
            )
            probs = classify(model, device, seq)
            buf_frames = buffer.frames()
            raw_index = int(np.argmax(probs))
            inferences.append({
                "t": round(t, 3),
                "probs": probs,
                "raw_label": labels[raw_index],
                "raw_p": round(float(probs[raw_index]), 4),
                "window_frames": len(buf_frames),
                "window_hand_rate": round(hand_coverage(buf_frames), 4),
                "window_duration": round(buffer.duration, 3),
            })
    return inferences, effective_start


def simulate_gate(
    inferences,
    labels,
    threshold,
    min_streak,
    alpha,
    min_hand_coverage=0.0,
    reset_after_missing_seconds=0.8,
    suppress_indices=None,
):
    """Re-simulate the demo DecisionGate from stored raw probabilities."""
    gate = DecisionGate(
        alpha=alpha,
        min_streak=min_streak,
        threshold=threshold,
        min_hand_coverage=min_hand_coverage,
        reset_after_missing_seconds=reset_after_missing_seconds,
        suppress_indices=suppress_indices,
    )
    out = []
    for item in inferences:
        decision = gate.update(
            item["probs"],
            coverage=item["window_hand_rate"],
            timestamp=item["t"],
        )
        out.append({
            "t": item["t"],
            "label": labels[decision.index] if decision.index is not None else None,
            "raw_label": labels[decision.raw_index],
            "raw_p": item["raw_p"],
            "confidence": round(float(decision.confidence), 4),
            "streak": decision.streak,
            "hand_coverage": item["window_hand_rate"],
            "blocked": decision.blocked,
            "suppressed": decision.suppressed,
            "reason": decision.reason,
        })
    return out


def summarise_gate(gated, target, labels):
    """Committed-prediction precision/coverage plus the settled label."""
    blocked = [g for g in gated if g["blocked"]]
    suppressed = [g for g in gated if g.get("suppressed")]
    committed = [g for g in gated if g["label"] is not None and not g["blocked"]]
    correct = [g for g in committed if g["label"] == target]
    counts: dict[str, int] = {}
    for g in committed:
        counts[g["label"]] = counts.get(g["label"], 0) + 1
    settled = max(counts, key=counts.get) if counts else None
    raw_counts: dict[str, int] = {}
    for g in gated:
        if not g["blocked"]:
            raw_counts[g["raw_label"]] = raw_counts.get(g["raw_label"], 0) + 1
    raw_settled = max(raw_counts, key=raw_counts.get) if raw_counts else None
    return {
        "inferences": len(gated),
        "blocked": len(blocked),
        "blocked_rate": round(len(blocked) / len(gated), 4) if gated else 0.0,
        "suppressed": len(suppressed),
        "suppressed_rate": round(len(suppressed) / len(gated), 4) if gated else 0.0,
        "committed": len(committed),
        "coverage": round(len(committed) / len(gated), 4) if gated else 0.0,
        "committed_correct": len(correct),
        "committed_precision": round(len(correct) / len(committed), 4) if committed else None,
        "settled_label": settled,
        "settled_correct": settled == target if settled else None,
        "raw_settled_label": raw_settled,
        "raw_settled_correct": raw_settled == target if raw_settled else None,
        "label_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def run_sweep(
    results,
    labels,
    *,
    alpha,
    reset_after_missing_seconds,
    thresholds,
    streaks,
    coverages,
    suppress_indices=None,
):
    """Grid-search gate settings on stored probabilities.

    Returns one row per (threshold, min_streak, min_hand_coverage) with pooled
    committed precision and settled top-1 over the labelled clips.
    """
    rows = []
    for threshold in thresholds:
        for min_streak in streaks:
            for coverage in coverages:
                settled_correct = 0
                committed = 0
                correct = 0
                blocked = 0
                total = 0
                for entry in results:
                    if "stream" not in entry:
                        continue
                    detail = entry["stream"]["inference_detail"]
                    inferences = [
                        {
                            "t": d["t"],
                            "probs": np.asarray(p, dtype=np.float32),
                            "raw_label": d["raw_label"],
                            "raw_p": d["raw_p"],
                            "window_hand_rate": d["window_hand_rate"],
                        }
                        for d, p in zip(detail, entry["_probs"])
                    ]
                    gated = simulate_gate(
                        inferences,
                        labels,
                        threshold,
                        min_streak,
                        alpha,
                        coverage,
                        reset_after_missing_seconds,
                        suppress_indices,
                    )
                    score = summarise_gate(gated, entry["target"], labels)
                    settled_correct += bool(score["settled_correct"])
                    committed += score["committed"]
                    correct += score["committed_correct"]
                    blocked += score["blocked"]
                    total += score["inferences"]
                rows.append({
                    "threshold": round(float(threshold), 3),
                    "min_streak": int(min_streak),
                    "min_hand_coverage": round(float(coverage), 3),
                    "settled_top1": round(settled_correct / len(results), 4) if results else 0.0,
                    "committed": committed,
                    "committed_correct": correct,
                    "committed_precision": round(correct / committed, 4) if committed else None,
                    "coverage": round(committed / total, 4) if total else 0.0,
                    "blocked_rate": round(blocked / total, 4) if total else 0.0,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--clips-dir", type=Path, default=Path("eval_vids"))
    parser.add_argument("--mode", choices=["clip", "stream", "both"], default="both")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--start-after", type=float, default=None)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-streak", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search threshold/min_streak/min_hand_coverage on stored probs",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        checkpoint_path = args.checkpoint or cfg.path("evaluation.checkpoint")
        if not checkpoint_path.is_file():
            raise ConfigError(f"Checkpoint not found: {checkpoint_path}")
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        )
        model, payload = load_checkpoint_model(checkpoint_path, device)
        labels: list[str] = list(payload["labels"])
        sign_hash = str(payload.get("sign_hash", ""))
        if sign_hash and sign_hash != cfg.sign_hash:
            raise ConfigError(
                f"Checkpoint sign_hash {sign_hash} != config {cfg.sign_hash}. Retrain first."
            )
        rest_index = labels.index(cfg.rest_label) if cfg.rest_label in labels else None
        suppress_labels = cfg.suppress_labels
        suppress_indices = {
            labels.index(label) for label in suppress_labels if label in labels
        }

        resample_frames = int(cfg.get("keypoints.resample_frames", 32))
        pose_points = int(cfg.get("keypoints.pose_points", 25))
        use_z = bool(cfg.get("keypoints.use_z", False))
        presence_flags = bool(cfg.get("keypoints.presence_flags", True))
        shoulder_visibility = float(cfg.get("keypoints.shoulder_visibility", 0.3))
        window = float(args.window if args.window is not None else cfg.get("demo.window_seconds", 2.5))
        start_after = float(args.start_after if args.start_after is not None else cfg.get("demo.start_after_seconds", 2.0))
        stride = float(args.stride if args.stride is not None else cfg.get("demo.infer_every_seconds", 0.33))
        alpha = float(args.alpha if args.alpha is not None else cfg.get("demo.smoothing_alpha", 0.5))
        min_streak = int(args.min_streak if args.min_streak is not None else cfg.get("demo.min_streak", 3))
        threshold = float(args.threshold if args.threshold is not None else cfg.get("demo.confidence_threshold", 0.6))
        min_hand_coverage = float(cfg.get("demo.min_hand_coverage", 0.0))
        reset_after_missing_seconds = float(
            cfg.get("demo.reset_after_missing_seconds", 0.8)
        )

        clips = sorted(args.clips_dir.expanduser().glob("*.mp4"))
        if not clips:
            raise ConfigError(f"No .mp4 files in {args.clips_dir}")
        label_to_index = {lab: i for i, lab in enumerate(labels)}

        results = []
        with Landmarkers.from_config(cfg) as landmarkers:
            for path in clips:
                target = normalise_label(path.stem)
                if target not in label_to_index:
                    print(f"SKIP {path.name}: '{target}' is not a config label", file=sys.stderr)
                    continue
                frames, fps, n_frames = read_video_frames(
                    landmarkers, path, shoulder_visibility=shoulder_visibility
                )
                entry = {
                    "clip": path.name,
                    "target": target,
                    "fps": round(fps, 2),
                    "duration_seconds": round(n_frames / fps, 2),
                    **frame_stats(frames),
                }
                if args.mode in ("clip", "both"):
                    entry["clip_mode"] = evaluate_clip(
                        model, device, frames, cfg, labels,
                        resample_frames, pose_points, use_z, presence_flags,
                        shoulder_visibility, rest_index,
                    )
                if args.mode in ("stream", "both"):
                    inferences, effective_start = stream_clip(
                        model, device, frames, fps, cfg, labels,
                        window=window, start_after=start_after, stride=stride,
                        alpha=alpha, resample_frames=resample_frames,
                        pose_points=pose_points, use_z=use_z,
                        presence_flags=presence_flags,
                        shoulder_visibility=shoulder_visibility,
                    )
                    entry["effective_start_seconds"] = round(effective_start, 3)
                    entry["stream"] = summarise_gate(
                        simulate_gate(
                            inferences,
                            labels,
                            threshold,
                            min_streak,
                            alpha,
                            min_hand_coverage,
                            reset_after_missing_seconds,
                            suppress_indices,
                        ),
                        target,
                        labels,
                    )
                    entry["stream"]["mean_window_hand_rate"] = (
                        round(float(np.mean([i["window_hand_rate"] for i in inferences])), 4)
                        if inferences else 0.0
                    )
                    entry["stream"]["inference_detail"] = [
                        {
                            "t": i["t"],
                            "raw_label": i["raw_label"],
                            "raw_p": i["raw_p"],
                            "window_hand_rate": i["window_hand_rate"],
                        }
                        for i in inferences
                    ]
                    # keep the probability vectors for the sweep, but do not bloat JSON
                    entry["_probs"] = [i["probs"].tolist() for i in inferences]
                results.append(entry)
                print(
                    f"{path.stem:<12} fps={fps:4.1f} hands={entry['any_hand_rate']*100:5.1f}% "
                    f"wrist_in={entry['wrist_in_frame_rate']*100:5.1f}% "
                    + (
                        f"clip={entry['clip_mode']['top1']:<10} p={entry['clip_mode']['top1_p']:.2f}"
                        if "clip_mode" in entry else ""
                    )
                    + (
                        f" stream_settled={entry['stream']['settled_label']} "
                        f"cov={entry['stream']['coverage']:.2f}"
                        if "stream" in entry else ""
                    )
                )

        # ---------------- aggregates ----------------
        report: dict = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint_path),
            "epoch": payload.get("epoch"),
            "clips_dir": str(args.clips_dir),
            "settings": {
                "window_seconds": window,
                "start_after_seconds": start_after,
                "stride_seconds": stride,
                "alpha": alpha,
                "min_streak": min_streak,
                "threshold": threshold,
                "min_hand_coverage": min_hand_coverage,
                "reset_after_missing_seconds": reset_after_missing_seconds,
                "suppress_labels": suppress_labels,
            },
            "clips": results,
        }

        if args.mode in ("clip", "both"):
            n = len(results)
            corr = sum(r["clip_mode"]["top1"] == r["target"] for r in results)
            top3 = sum(r["target"] in [t["label"] for t in r["clip_mode"]["top3"]] for r in results)
            report["clip_accuracy"] = {
                "n": n,
                "top1": round(corr / n, 4),
                "top3": round(top3 / n, 4),
                "errors": [
                    {"target": r["target"], "pred": r["clip_mode"]["top1"], "p": r["clip_mode"]["top1_p"]}
                    for r in results if r["clip_mode"]["top1"] != r["target"]
                ],
            }

        if args.mode in ("stream", "both"):
            n = len(results)
            settled = sum(bool(r["stream"]["settled_correct"]) for r in results)
            raw_settled = sum(bool(r["stream"]["raw_settled_correct"]) for r in results)
            committed = sum(r["stream"]["committed"] for r in results)
            correct = sum(r["stream"]["committed_correct"] for r in results)
            blocked = sum(r["stream"]["blocked"] for r in results)
            suppressed = sum(r["stream"].get("suppressed", 0) for r in results)
            total = sum(r["stream"]["inferences"] for r in results)
            report["stream_score"] = {
                "n": n,
                "settled_top1": round(settled / n, 4),
                "raw_settled_top1": round(raw_settled / n, 4),
                "committed_total": committed,
                "committed_correct": correct,
                "committed_precision": round(correct / committed, 4) if committed else None,
                "coverage": round(committed / total, 4) if total else 0.0,
                "blocked_rate": round(blocked / total, 4) if total else 0.0,
                "suppressed_rate": round(suppressed / total, 4) if total else 0.0,
                "mean_any_hand_rate": round(float(np.mean([r["any_hand_rate"] for r in results])), 4),
                "mean_window_hand_rate": round(float(np.mean([r["stream"]["mean_window_hand_rate"] for r in results])), 4),
            }

            if args.sweep:
                rows = run_sweep(
                    results,
                    labels,
                    alpha=alpha,
                    reset_after_missing_seconds=reset_after_missing_seconds,
                    thresholds=[0.4, 0.5, 0.6, 0.7, 0.8],
                    streaks=[2, 3, 4],
                    coverages=[0.0, 0.2, 0.3, 0.4, 0.5],
                    suppress_indices=suppress_indices,
                )
                report["sweep"] = rows
                ranked = sorted(
                    rows,
                    key=lambda r: (
                        r["settled_top1"],
                        r["committed_precision"] or 0.0,
                        r["coverage"],
                    ),
                    reverse=True,
                )
                print()
                print("Top gate settings (settled top-1, then precision, then coverage):")
                print(f"  {'thr':>4} {'streak':>6} {'mincov':>6} {'settled':>8} "
                      f"{'prec':>6} {'cov':>6} {'blocked':>7}")
                for r in ranked[:10]:
                    print(f"  {r['threshold']:>4.1f} {r['min_streak']:>6d} "
                          f"{r['min_hand_coverage']:>6.2f} "
                          f"{r['settled_top1']*100:>7.1f}% "
                          f"{(r['committed_precision'] or 0)*100:>5.1f}% "
                          f"{r['coverage']*100:>5.1f}% {r['blocked_rate']*100:>6.1f}%")

        for entry in results:
            entry.pop("_probs", None)

        out_path = args.out or (
            cfg.path("evaluation.reports_dir")
            / f"live_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = json.loads(json.dumps(report, default=float))
        out_path.write_text(json.dumps(serialisable, indent=1), encoding="utf-8")

        print()
        print("=" * 78)
        if "clip_accuracy" in report:
            c = report["clip_accuracy"]
            print(f"CLIP MODE (oracle segmentation): top1 {c['top1']*100:.1f}%  top3 {c['top3']*100:.1f}%  n={c['n']}")
            for e in c["errors"]:
                print(f"  miss: {e['target']:<12} -> {e['pred']:<12} p={e['p']:.2f}")
        if "stream_score" in report:
            s = report["stream_score"]
            print(f"STREAM MODE: settled top1 {s['settled_top1']*100:.1f}%  "
                  f"committed precision {s['committed_precision']}  "
                  f"coverage {s['coverage']*100:.1f}%  "
                  f"blocked {s['blocked_rate']*100:.1f}%  "
                  f"suppressed {s['suppressed_rate']*100:.1f}%")
            print(f"  gate: threshold {threshold:.2f}, min streak {min_streak}, "
                  f"min hand coverage {min_hand_coverage:.2f}")
            print(f"  mean hand presence in windows: {s['mean_window_hand_rate']*100:.1f}%")
        print(f"Report: {out_path}")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
