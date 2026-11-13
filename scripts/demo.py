#!/usr/bin/env python3
"""Live webcam ISL word recognition demo.

Pipeline per frame: MediaPipe pose + both hands -> shoulder-normalized frame ->
2.5 s ring buffer -> 32-frame resample -> BiGRU -> EMA smoothing + streak gate.

Controls: press ``q`` or ``Esc`` to quit.

Examples:
    python scripts/demo.py                       # default webcam
    python scripts/demo.py --source 1            # second camera
    python scripts/demo.py --source clip.MOV     # test on a video file
    python scripts/demo.py --headless --max-seconds 20 --source clip.MOV

The session summary (FPS, low-confidence rate, frames without hands, label
switches) is printed at exit and written to ``reports/demo_session_*.json`` for
the separate live-demo failure analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from _bootstrap import ROOT, add_config_args

from isl.config import ConfigError, load_config
from isl.keypoints import Landmarkers, detect_frame
from isl.live import (
    DecisionGate,
    DemoStats,
    TimedFrameBuffer,
    buffer_to_sequence,
    hand_coverage,
    wrist_out_of_frame,
)
from isl.model import load_checkpoint_model


def parse_source(text: str) -> int | str:
    return int(text) if text.strip().isdigit() else str(Path(text).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--source", default=None, help="Camera index or video path")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window", type=float, default=None, help="Seconds in buffer")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-streak", type=int, default=None)
    parser.add_argument(
        "--min-hand-coverage",
        type=float,
        default=None,
        help="Minimum fraction of window frames with a detected hand before committing",
    )
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument(
        "--start-after",
        type=float,
        default=None,
        help="Seconds of buffered video before the first prediction",
    )
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 = unlimited")
    parser.add_argument("--headless", action="store_true", help="No OpenCV window")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera image")
    parser.add_argument("--loop", action="store_true", help="Loop a video file")
    parser.add_argument(
        "--session-out",
        type=Path,
        default=None,
        help="Session JSON (default: reports/demo_session_<time>.json)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        checkpoint_path = args.checkpoint or cfg.path("evaluation.checkpoint")
        if not checkpoint_path.is_file():
            raise ConfigError(
                f"Checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
            )
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        model, payload = load_checkpoint_model(checkpoint_path, device)
        labels: list[str] = list(payload["labels"])
        sign_hash = str(payload.get("sign_hash", ""))
        if sign_hash and sign_hash != cfg.sign_hash:
            raise ConfigError(
                f"Checkpoint sign_hash {sign_hash} != config {cfg.sign_hash}. Retrain first."
            )
        window_seconds = float(
            args.window if args.window is not None else cfg.get("demo.window_seconds", 2.5)
        )
        threshold = float(
            args.threshold
            if args.threshold is not None
            else cfg.get("demo.confidence_threshold", 0.6)
        )
        alpha = float(
            args.alpha
            if args.alpha is not None
            else cfg.get("demo.smoothing_alpha", 0.5)
        )
        min_streak = int(
            args.min_streak
            if args.min_streak is not None
            else cfg.get("demo.min_streak", 3)
        )
        min_hand_coverage = float(
            args.min_hand_coverage
            if args.min_hand_coverage is not None
            else cfg.get("demo.min_hand_coverage", 0.0)
        )
        start_after = float(
            args.start_after
            if args.start_after is not None
            else cfg.get("demo.start_after_seconds", min(2.0, window_seconds))
        )
        infer_every = float(cfg.get("demo.infer_every_seconds", 0.33))
        resample_frames = int(cfg.get("keypoints.resample_frames", 32))
        pose_points = int(cfg.get("keypoints.pose_points", 25))
        use_z = bool(cfg.get("keypoints.use_z", False))
        presence_flags = bool(cfg.get("keypoints.presence_flags", True))
        shoulder_visibility = float(cfg.get("keypoints.shoulder_visibility", 0.3))

        rest_index = labels.index(cfg.rest_label) if cfg.rest_label in labels else None
        rest_display = cfg.display(cfg.rest_label) if rest_index is not None else "No sign"
        suppress_labels = cfg.suppress_labels
        suppress_indices = {
            labels.index(label) for label in suppress_labels if label in labels
        }

        source = parse_source(
            args.source
            if args.source is not None
            else str(cfg.get("demo.camera", 0))
        )
        is_camera = isinstance(source, int)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConfigError(f"Could not open video source: {source!r}")
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("demo.width", 640)))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("demo.height", 480)))

        landmarkers = Landmarkers.from_config(cfg)
        buffer = TimedFrameBuffer(window_seconds)
        gate = DecisionGate(
            alpha=alpha,
            min_streak=min_streak,
            threshold=threshold,
            min_hand_coverage=min_hand_coverage,
            reset_after_missing_seconds=float(
                cfg.get("demo.reset_after_missing_seconds", 0.8)
            ),
            suppress_indices=suppress_indices,
        )
        stats = DemoStats()

        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_index = 0
        started = time.perf_counter()
        last_ts = -1
        last_inference = -1e9
        predicted_display = "..."
        display_active = False
        confidence = 0.0
        fps = 0.0
        last_frame_time = started

        print(f"checkpoint : {checkpoint_path} (epoch {payload.get('epoch')})")
        print(f"source     : {source!r}")
        print(f"device     : {device}")
        print(f"window     : {window_seconds:.1f} s (first prediction after "
              f"{start_after:.1f} s) | threshold {threshold:.2f} | "
              f"alpha {alpha:.2f} | min streak {min_streak} | "
              f"min hand coverage {min_hand_coverage:.2f}")
        if suppress_labels:
            print(f"suppressed : {', '.join(suppress_labels)} (model still predicts, demo hides)")
        print("Press q or Esc to quit." if not args.headless else "Headless mode.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if args.loop and not is_camera:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                now = time.perf_counter()
                elapsed = now - started
                if args.max_seconds and elapsed > args.max_seconds:
                    break

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0:
                    instant = 1.0 / dt
                    fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant

                # Video files may be processed faster than real time, so use the
                # clip's own timebase there. A camera is live, so wall clock is
                # the correct timebase (and handles dropped frames).
                if not is_camera and video_fps > 0:
                    stream_time = frame_index / video_fps
                else:
                    stream_time = elapsed
                frame_index += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(stream_time * 1000)
                if timestamp <= last_ts:
                    timestamp = last_ts + 1
                last_ts = timestamp

                raw = detect_frame(
                    landmarkers, image, timestamp, shoulder_visibility
                )
                buffer.append(stream_time, raw)
                stats.observe_frame(raw)

                if (
                    len(buffer) >= max(2, args.min_frames)
                    and buffer.duration >= start_after
                    and stream_time - last_inference >= infer_every
                ):
                    last_inference = stream_time
                    sequence = buffer_to_sequence(
                        buffer.frames(),
                        resample_frames=resample_frames,
                        pose_points=pose_points,
                        use_z=use_z,
                        presence_flags=presence_flags,
                        shoulder_visibility=shoulder_visibility,
                    )
                    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

                    window_frames = buffer.frames()
                    coverage = hand_coverage(window_frames)
                    decision = gate.update(
                        probs, coverage=coverage, timestamp=stream_time
                    )
                    raw_display = cfg.display(labels[decision.raw_index])
                    confidence = decision.confidence
                    order = np.argsort(probs)[::-1]
                    top3 = [
                        {
                            "label": labels[int(i)],
                            "p": round(float(probs[int(i)]), 4),
                        }
                        for i in order[:3]
                    ]
                    margin = float(probs[order[0]] - probs[order[1]])
                    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

                    if decision.blocked:
                        predicted_display = "Hands out of frame"
                    elif decision.suppressed:
                        predicted_display = "..."
                    elif decision.index is None:
                        predicted_display = "..."
                    elif decision.index == rest_index:
                        predicted_display = rest_display
                    else:
                        predicted_display = cfg.display(labels[decision.index])
                    display_active = decision.index is not None

                    stats.observe_prediction(
                        timestamp=stream_time,
                        label=(
                            None
                            if decision.index is None
                            else labels[decision.index]
                        ),
                        display=predicted_display,
                        confidence=decision.confidence,
                        raw_label=labels[decision.raw_index],
                        raw_confidence=decision.raw_confidence,
                        streak=decision.streak,
                        fps=fps,
                        rest_label=cfg.rest_label if rest_index is not None else None,
                        threshold=threshold,
                        hand_coverage=coverage,
                        blocked=decision.blocked,
                        suppressed=decision.suppressed,
                        reason=decision.reason,
                        top3=top3,
                        margin=margin,
                        entropy=entropy,
                    )

                    if args.headless:
                        print(
                            f"{stream_time:6.2f}s  {predicted_display:<22} "
                            f"conf {decision.confidence:.3f}  "
                            f"raw {raw_display:<18} {decision.raw_confidence:.3f}  "
                            f"hands {coverage:.2f}  fps {fps:5.1f}"
                        )

                if not args.headless:
                    colour = (
                        (60, 200, 60) if display_active else (120, 120, 120)
                    )
                    cv2.putText(
                        frame,
                        predicted_display,
                        (18, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        colour,
                        3,
                        cv2.LINE_AA,
                    )
                    framing = (
                        "hands out of frame - step back"
                        if wrist_out_of_frame(buffer.frames())
                        else f"conf {confidence:.2f}  streak {gate.smoother.streak}"
                    )
                    cv2.putText(
                        frame,
                        framing,
                        (18, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        colour,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"{fps:5.1f} FPS  buffer {buffer.duration:.1f}s",
                        (18, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (230, 230, 230),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("INCLUDE-50 live ISL", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            landmarkers.close()
            if not args.headless:
                cv2.destroyAllWindows()

        duration = time.perf_counter() - started
        summary = stats.summary(duration)
        print()
        print("=" * 70)
        print(f"Session: {duration:.1f}s, {summary['frames']} frames, "
              f"{summary['average_fps']} FPS, {summary['inferences']} inferences")
        print(f"Frames with no hands : {summary['frames_without_hands']}")
        print(f"Low-confidence       : {summary['low_confidence_inferences']} "
              f"of {summary['inferences']} inferences")
        print(f"Suppressed commits   : {summary['suppressed_inferences']}"
              + (f" ({', '.join(suppress_labels)})" if suppress_labels else ""))
        print(f"Label switches       : {summary['label_switches']}")
        if summary["label_time"]:
            print("Labels seen          :")
            for label, count in summary["label_time"].items():
                print(f"  {label:<20} {count}")

        session_path = args.session_out or (
            cfg.path("evaluation.reports_dir")
            / f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": str(checkpoint_path),
                    "source": str(source),
                    "device": str(device),
                    "settings": {
                        "window_seconds": window_seconds,
                        "start_after_seconds": start_after,
                        "threshold": threshold,
                        "alpha": alpha,
                        "min_streak": min_streak,
                        "min_hand_coverage": min_hand_coverage,
                        "reset_after_missing_seconds": float(
                            cfg.get("demo.reset_after_missing_seconds", 0.8)
                        ),
                        "infer_every_seconds": infer_every,
                        "mirror": args.mirror,
                        "suppress_labels": suppress_labels,
                    },
                    "summary": summary,
                    "timeline": stats.timeline,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print()
        print(f"Session log: {session_path}")
        print("Use this log for the live-demo failure analysis in the report.")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
