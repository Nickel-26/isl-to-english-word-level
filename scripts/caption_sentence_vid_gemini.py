#!/usr/bin/env python3
"""Render a video captioned with FINAL rewritten sentences only.

This is a separate, offline script. It never draws glosses, gate decisions,
warnings or any debug text into the output video -- just the finished English
sentence in the caption bar, like a normal subtitled video.

It makes two passes over the input file:

  1. Run the exact same MediaPipe -> BiGRU -> DecisionGate -> UtteranceBuffer
     path as ``scripts/sentence_demo.py``. Committed glosses are rewritten by
     Gemini in a background worker, so a slow API call never stalls the decoder.
  2. Re-read the file and burn only the resulting sentences into a new silent
     video. A sentence starts when its first sign was committed and stays until
     the next utterance starts; the last one is held for ``--hold-seconds``.

The classifier stays word-level and is not retrained. A JSON sidecar stores the
final sentence timings only (no glosses, no API key).

Examples:
    python scripts/caption_sentence_vid.py eval_vids/cow.mp4
    python scripts/caption_sentence_vid.py clip.mp4 --out /tmp/final.mp4
    python scripts/caption_sentence_vid.py clip.mp4 --echo  # offline debug only
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

from _bootstrap import ROOT, add_config_args

from isl.captions import draw_caption_bar, sentence_at, wrap_caption
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
from isl.sentence import (
    AsyncRewriter,
    EchoRewriter,
    GeminiRewriter,
    UtteranceBuffer,
    read_env_key,
)


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
        help="Output video (default: reports/<name>_sentence.mp4)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Sentence-timing JSON (default: alongside the output video)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")

    # Same gate overrides as the other video scripts.
    parser.add_argument("--window", type=float, default=None, help="Seconds in buffer")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-streak", type=int, default=None)
    parser.add_argument("--min-hand-coverage", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--start-after", type=float, default=None)
    parser.add_argument(
        "--max-seconds", type=float, default=0.0, help="Stop after N seconds (0 = all)"
    )

    # Sentence-layer overrides (the shipped config is the source of truth).
    parser.add_argument("--gap", type=float, default=None)
    parser.add_argument("--max-glosses", type=int, default=None)
    parser.add_argument("--max-wait", type=float, default=None)
    parser.add_argument("--model", default=None, help="Override sentence.gemini_model")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f"Env file for the key (default: {ROOT / '.env'})",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Offline plumbing debug: captions glosses; NOT a sentence rewrite",
    )

    # Rendering only.
    parser.add_argument(
        "--font-scale", type=float, default=1.0, help="Multiplier on caption text size"
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="Hold the last sentence on extra frames after the video ends",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser()
    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        return 1
    if args.hold_seconds < 0:
        print("ERROR: --hold-seconds must be >= 0", file=sys.stderr)
        return 1

    try:
        cfg = load_config(args.config, root=args.root)
        settings = cfg.sentence
        api_key_env = args.api_key_env or settings.api_key_env
        api_key = (
            ""
            if args.echo
            else read_env_key(api_key_env, env_file=args.env_file, root=ROOT)
        )

        gap_seconds = float(args.gap if args.gap is not None else settings.gap_seconds)
        max_glosses = int(
            args.max_glosses if args.max_glosses is not None else settings.max_glosses
        )
        max_wait = float(
            args.max_wait if args.max_wait is not None else settings.max_wait_seconds
        )
        gemini_model = args.model or settings.gemini_model
        timeout_seconds = float(
            args.timeout if args.timeout is not None else settings.timeout_seconds
        )
        if not args.echo and not api_key:
            raise ConfigError(
                f"No API key in ${api_key_env} or {ROOT / '.env'}. Export it or add "
                "it to .env. Use --echo for an offline plumbing check. The key is "
                "never written to a log."
            )

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
            args.window if args.window is not None else cfg.get("demo.window_seconds", 2.5)
        )
        threshold = float(
            args.threshold
            if args.threshold is not None
            else cfg.get("demo.confidence_threshold", 0.6)
        )
        alpha = float(
            args.alpha if args.alpha is not None else cfg.get("demo.smoothing_alpha", 0.5)
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
        suppress_indices = {
            labels.index(label) for label in cfg.suppress_labels if label in labels
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
        effective_start = start_after
        if video_duration > 0:
            effective_start = min(start_after, max(0.5 * video_duration, 0.35))

        out_path = (
            args.out
            or cfg.path("evaluation.reports_dir")
            / f"{input_path.stem}_sentence.mp4"
        ).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open a video writer for {out_path} (codec mp4v)")

        font_scale = max(0.45, height / 720.0 * 0.85) * max(0.1, args.font_scale)
        thickness = max(1, int(round(font_scale * 2)))
        pad_x = max(8, int(round(width * 0.02)))
        max_text_width = max(32, width - 2 * pad_x)

        def caption_lines(sentence: str) -> list[str]:
            return (
                wrap_caption(
                    sentence.split(),
                    font_scale=font_scale,
                    thickness=thickness,
                    max_width=max_text_width,
                    max_lines=2,
                )
                if sentence
                else []
            )

        print(f"input      : {input_path}  ({width}x{height}, {video_fps:.2f} fps)")
        print(f"checkpoint : {checkpoint_path} (epoch {payload.get('epoch')})")
        print(f"output     : {out_path}")
        print(
            f"gate       : window {window_seconds:.1f}s | threshold {threshold:.2f} | "
            f"streak {min_streak} | min hand coverage {min_hand_coverage:.2f}"
        )
        print(
            f"sentence   : gap {gap_seconds:.1f}s | max {max_glosses} glosses / "
            f"{max_wait:.1f}s | backend "
            f"{'echo (offline debug)' if args.echo else gemini_model}"
        )
        print("Analysing signs and rewriting sentences...")

        # ------------------------------------------------------------------
        # Pass 1: recognition + background rewrite. No frames are written.
        # ------------------------------------------------------------------
        if args.echo:
            rewriter = EchoRewriter()
        else:
            rewriter = GeminiRewriter(
                api_key=api_key,
                model=gemini_model,
                timeout_seconds=timeout_seconds,
                temperature=settings.temperature,
            )
        runner = AsyncRewriter(rewriter)
        started = time.perf_counter()
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
        utterances = UtteranceBuffer(
            gap_seconds=gap_seconds,
            max_glosses=max_glosses,
            max_wait_seconds=max_wait,
        )
        stats = DemoStats()
        pending = first_frame
        frame_index = 0
        last_ts = -1
        last_inference = -1e9
        stream_time = 0.0
        results = []

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

                raw = detect_frame(landmarkers, image, timestamp, shoulder_visibility)
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

                    if decision.blocked or decision.index is None or decision.suppressed:
                        completed = utterances.on_no_sign(stream_time)
                    elif decision.index == rest_index:
                        completed = utterances.on_rest(stream_time)
                    else:
                        label = labels[decision.index]
                        completed = utterances.push(
                            label, cfg.display(label), stream_time
                        )
                    if completed is not None and len(completed):
                        runner.submit(completed, stream_time)

                    stats.observe_prediction(
                        timestamp=stream_time,
                        label=(
                            None if decision.index is None else labels[decision.index]
                        ),
                        display=(
                            "..."
                            if decision.index is None
                            else cfg.display(labels[decision.index])
                        ),
                        confidence=decision.confidence,
                        raw_label=labels[decision.raw_index],
                        raw_confidence=decision.raw_confidence,
                        streak=decision.streak,
                        fps=video_fps,
                        rest_label=cfg.rest_label if rest_index is not None else None,
                        threshold=threshold,
                        hand_coverage=coverage,
                        blocked=decision.blocked,
                        suppressed=decision.suppressed,
                        reason=decision.reason,
                    )

            final = utterances.flush("eof", stream_time)
            if final is not None and len(final):
                runner.submit(final, stream_time)
            results = runner.close()  # offline: wait for every pending rewrite
        finally:
            cap.release()
            landmarkers.close()

        # Start each caption when its first sign was committed. An empty text
        # clears the bar so a failed rewrite never leaves a stale sentence.
        entries: list[tuple[float, str]] = []
        for utterance, _finished_at, result in results:
            text = result.text.strip() if result.ok else ""
            entries.append((float(utterance.started_at), text))
        entries.sort(key=lambda item: item[0])
        final_sentences = [text for _start, text in entries if text]
        failed = len(entries) - len(final_sentences)

        # ------------------------------------------------------------------
        # Pass 2: re-read the file and draw ONLY the final sentences.
        # ------------------------------------------------------------------
        print(f"Rendering {len(final_sentences)} final sentence(s)...")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not re-open video: {input_path}")
        written = 0
        last_frame = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = written / video_fps
                written += 1
                if args.max_seconds and t > args.max_seconds:
                    break
                lines = caption_lines(sentence_at(entries, t))
                draw_caption_bar(frame, lines, font_scale=font_scale, thickness=thickness)
                writer.write(frame)
                last_frame = frame.copy()

            if (
                entries
                and entries[-1][1]
                and last_frame is not None
                and args.hold_seconds > 0
            ):
                lines = caption_lines(entries[-1][1])
                extra = int(round(args.hold_seconds * video_fps))
                for _ in range(max(0, extra)):
                    held = last_frame.copy()
                    draw_caption_bar(
                        held, lines, font_scale=font_scale, thickness=thickness
                    )
                    writer.write(held)
                    written += 1
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - started
        raw_summary = stats.summary(elapsed)
        # Keep the sidecar to the final product: no per-frame gloss timeline.
        summary = {
            "duration_seconds": raw_summary.get("duration_seconds"),
            "frames": raw_summary.get("frames"),
            "inferences": raw_summary.get("inferences"),
            "video_fps": round(video_fps, 3),
            "frames_written": written,
            "sentences": len(final_sentences),
            "failed_rewrites": failed,
        }

        json_path = (args.json_out or out_path.with_suffix(".json")).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "input": str(input_path),
                    "output": str(out_path),
                    "checkpoint": str(checkpoint_path),
                    "backend": "echo" if args.echo else gemini_model,
                    "settings": {
                        "window_seconds": window_seconds,
                        "start_after_seconds": effective_start,
                        "threshold": threshold,
                        "alpha": alpha,
                        "min_streak": min_streak,
                        "min_hand_coverage": min_hand_coverage,
                        "gap_seconds": gap_seconds,
                        "max_glosses": max_glosses,
                        "max_wait_seconds": max_wait,
                        "timeout_seconds": timeout_seconds,
                        "hold_seconds": args.hold_seconds,
                        "suppressed_labels": cfg.suppress_labels,
                    },
                    "summary": summary,
                    "sentences": [
                        {"start": round(start, 3), "text": text}
                        for start, text in entries
                        if text
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 70)
        print(f"Frames written : {written}")
        print(f"Sentences      : {len(final_sentences)}")
        for start, text in entries:
            print(f"  {start:6.2f}s  {text or '(rewrite failed)'}")
        print(f"Output video   : {out_path}")
        print(f"Timing JSON    : {json_path}")
        return 0
    except (ConfigError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
