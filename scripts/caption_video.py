#!/usr/bin/env python3
"""Render a video with recognised ISL words burned in as captions.

Same inference path as the live demo (MediaPipe pose + hands, 2.5 s window,
32-frame resample, BiGRU, hand-coverage + EMA + threshold + streak gate), but
the output is a new video file instead of an OpenCV window. A word is added to
a rolling caption line only when the gate commits it; the synthetic ``rest``
class is rendered as a full stop.

Examples:
    python scripts/caption_video.py eval_vids/cow.mp4
    python scripts/caption_video.py clip.mp4 --out /tmp/captioned.mp4 --threshold 0.7

The output video is silent: OpenCV's VideoWriter has no audio track. A JSON
sidecar next to the video records the transcript and every inference decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from _bootstrap import add_config_args

from isl.captions import CaptionState, draw_caption_bar, wrap_caption
from isl.config import ConfigError, load_config
from isl.keypoints import Landmarkers, detect_frame
from isl.live import (
    DecisionGate,
    DemoStats,
    TimedFrameBuffer,
    buffer_to_sequence,
    hand_coverage,
)
from isl.model import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output video (default: reports/<name>_captions.mp4)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window", type=float, default=None, help="Seconds in buffer")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-streak", type=int, default=None)
    parser.add_argument("--min-hand-coverage", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument(
        "--start-after",
        type=float,
        default=None,
        help="Seconds of buffered video before the first prediction",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0, help="Stop after N seconds (0 = all)"
    )
    parser.add_argument(
        "--font-scale", type=float, default=1.0, help="Multiplier on caption text size"
    )
    parser.add_argument("--max-words", type=int, default=12)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Decision log (default: alongside the output video)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser()
    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(args.config, root=args.root)
        checkpoint_path = args.checkpoint or cfg.path("evaluation.checkpoint")
        if not checkpoint_path.is_file():
            raise ConfigError(
                f"Checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
            )
        device = torch.device(
            args.device
            if args.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model, payload = load_checkpoint_model(checkpoint_path, device)
        labels: list[str] = list(payload["labels"])
        sign_hash = str(payload.get("sign_hash", ""))
        if sign_hash and sign_hash != cfg.sign_hash:
            raise ConfigError(
                f"Checkpoint sign_hash {sign_hash} != config {cfg.sign_hash}. Retrain first."
            )

        window_seconds = float(
            args.window
            if args.window is not None
            else cfg.get("demo.window_seconds", 2.5)
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
            else cfg.get("demo.min_streak", 2)
        )
        min_hand_coverage = float(
            args.min_hand_coverage
            if args.min_hand_coverage is not None
            else cfg.get("demo.min_hand_coverage", 0.0)
        )
        reset_after_missing = float(cfg.get("demo.reset_after_missing_seconds", 0.8))
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
        suppress_labels = cfg.suppress_labels
        suppress_indices = {
            labels.index(label) for label in suppress_labels if label in labels
        }

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError(f"Could not decode any frame from: {input_path}")

        height, width = first_frame.shape[:2]
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        video_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 1.0 else 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_duration = frame_count / video_fps if frame_count > 0 else 0.0

        # A short take can be shorter than start_after; adapt so the first
        # prediction still happens instead of producing a blank video.
        effective_start = start_after
        if video_duration > 0:
            effective_start = min(
                start_after, max(0.5 * video_duration, 0.35)
            )

        out_path = args.out or (
            cfg.path("evaluation.reports_dir")
            / f"{input_path.stem}_captions.mp4"
        )
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open a video writer for {out_path} (codec mp4v)"
            )

        print(f"input      : {input_path}  ({width}x{height}, {video_fps:.2f} fps)")
        print(f"checkpoint : {checkpoint_path} (epoch {payload.get('epoch')})")
        print(f"output     : {out_path}")
        print(f"gate       : window {window_seconds:.1f}s | threshold {threshold:.2f} | "
              f"alpha {alpha:.2f} | streak {min_streak} | "
              f"min hand coverage {min_hand_coverage:.2f}")
        if suppress_labels:
            print(f"suppressed : {', '.join(suppress_labels)} (never captioned)")
        print("Captioning...")

        landmarkers = Landmarkers.from_config(cfg)
        buffer = TimedFrameBuffer(window_seconds)
        gate = DecisionGate(
            alpha=alpha,
            min_streak=min_streak,
            threshold=threshold,
            min_hand_coverage=min_hand_coverage,
            reset_after_missing_seconds=reset_after_missing,
            suppress_indices=suppress_indices,
        )
        captions = CaptionState(max_words=args.max_words)
        stats = DemoStats()
        caption_events: list[dict] = []

        font_scale = max(0.45, height / 720.0 * 0.85) * max(0.1, args.font_scale)
        thickness = max(1, int(round(font_scale * 2)))
        pad_x = max(8, int(round(width * 0.02)))
        caption_lines: list[str] = []

        pending = first_frame
        frame_index = 0
        last_ts = -1
        last_inference = -1e9
        started = time.perf_counter()

        try:
            while True:
                if pending is not None:
                    frame = pending
                    pending = None
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                stream_time = frame_index / video_fps
                frame_index += 1
                if args.max_seconds and stream_time > args.max_seconds:
                    break

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
                    and buffer.duration >= effective_start
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

                    coverage = hand_coverage(buffer.frames())
                    decision = gate.update(
                        probs, coverage=coverage, timestamp=stream_time
                    )
                    order = np.argsort(probs)[::-1]
                    top3 = [
                        {"label": labels[int(i)], "p": round(float(probs[int(i)]), 4)}
                        for i in order[:3]
                    ]
                    rest_label = cfg.rest_label if rest_index is not None else None
                    display = (
                        "Hands out of frame"
                        if decision.blocked
                        else (
                            "..."
                            if decision.index is None
                            else cfg.display(labels[decision.index])
                        )
                    )
                    stats.observe_prediction(
                        timestamp=stream_time,
                        label=(
                            None if decision.index is None else labels[decision.index]
                        ),
                        display=display,
                        confidence=decision.confidence,
                        raw_label=labels[decision.raw_index],
                        raw_confidence=decision.raw_confidence,
                        streak=decision.streak,
                        fps=video_fps,
                        rest_label=rest_label,
                        threshold=threshold,
                        hand_coverage=coverage,
                        blocked=decision.blocked,
                        suppressed=decision.suppressed,
                        reason=decision.reason,
                        top3=top3,
                        margin=float(probs[order[0]] - probs[order[1]]),
                        entropy=float(-(probs * np.log(probs + 1e-12)).sum()),
                    )

                    if decision.index is not None:
                        label = labels[decision.index]
                        is_rest = decision.index == rest_index
                        word = "." if is_rest else cfg.display(label)
                        if captions.push(
                            label,
                            cfg.display(label),
                            stream_time,
                            is_rest=is_rest,
                        ):
                            caption_events.append(
                                {"t": round(stream_time, 3), "word": word}
                            )
                            print(f"  {stream_time:6.2f}s  {word}")
                        caption_lines = wrap_caption(
                            captions.words,
                            font_scale=font_scale,
                            thickness=thickness,
                            max_width=max(32, width - 2 * pad_x),
                            max_lines=2,
                        )

                draw_caption_bar(
                    frame,
                    caption_lines,
                    font_scale=font_scale,
                    thickness=thickness,
                )
                writer.write(frame)
        finally:
            cap.release()
            writer.release()
            landmarkers.close()

        elapsed = time.perf_counter() - started
        summary = stats.summary(elapsed)
        summary["video_fps"] = round(video_fps, 3)
        summary["processed_seconds"] = round(frame_index / video_fps, 3)
        summary["video_duration_seconds"] = round(video_duration, 3)

        json_path = args.json_out or out_path.with_suffix(".json")
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "input": str(input_path),
                    "output": str(out_path),
                    "checkpoint": str(checkpoint_path),
                    "device": str(device),
                    "settings": {
                        "window_seconds": window_seconds,
                        "start_after_seconds": effective_start,
                        "threshold": threshold,
                        "alpha": alpha,
                        "min_streak": min_streak,
                        "min_hand_coverage": min_hand_coverage,
                        "reset_after_missing_seconds": reset_after_missing,
                        "infer_every_seconds": infer_every,
                        "max_words": args.max_words,
                        "suppress_labels": suppress_labels,
                    },
                    "summary": summary,
                    "transcript": captions.text,
                    "caption_events": caption_events,
                    "timeline": stats.timeline,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 70)
        print(f"Frames written : {frame_index}")
        print(f"Inferences     : {summary['inferences']}")
        if summary.get("suppressed_inferences"):
            print(f"Suppressed     : {summary['suppressed_inferences']} "
                  f"({', '.join(suppress_labels)})")
        print(f"Transcript     : {captions.text or '(none)'}")
        print(f"Output video   : {out_path}")
        print(f"Decision log   : {json_path}")
        if not caption_events:
            print("No committed words. Check framing: hands must stay in view.")
        return 0
    except (ConfigError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
