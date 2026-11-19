#!/usr/bin/env python3
"""EXPERIMENTAL Ollama sentence demo: committed glosses -> English sentence.

This is a *separate* webcam script, identical to ``scripts/sentence_demo.py``
except that the rewrite backend is a local Ollama model instead of Gemini. The
classifier stays word-level: one isolated sign, one prediction. The existing
decision gate's committed labels are buffered, and when signing stops (a
committed "No sign" or a silence gap) the glosses go to ``qwen2.5:3b-instruct``.

    big cow         -> "the cow is big"
    loud paint bird -> "the bird is loud"     (stray "paint" dropped)

No API key, internet access or API bill: Ollama must be running locally with
the model already pulled:

    ollama pull qwen2.5:3b-instruct
    ollama serve                      # if it is not already running
    python scripts/sentence_demo_ollama.py

No sentence training, no self-recorded training data, and no change to the
checkpoint's 97.92% test accuracy. The Gemini scripts are untouched.

Controls: press ``q`` or ``Esc`` to quit. Use ``--echo`` for an offline plumbing
check (echoes the gloss order; no grammar and no Ollama call).
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

from _bootstrap import add_config_args

from isl.captions import draw_caption_bar, wrap_caption
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
from isl.sentence import (
    AsyncRewriter,
    EchoRewriter,
    OllamaRewriter,
    RewriteResult,
    Utterance,
    UtteranceBuffer,
)


def parse_source(text: str) -> int | str:
    return int(text) if text.strip().isdigit() else str(Path(text).expanduser())


def _display_lines(gloss_text: str, last_sentence: str, hint: str) -> list[str]:
    lines: list[str] = []
    if last_sentence:
        lines.append(last_sentence)
    if gloss_text:
        lines.append(f"glosses: {gloss_text}")
    if hint:
        lines.append(hint)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--source", default=None, help="Camera index or video path")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window", type=float, default=None, help="Seconds in buffer")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--min-streak", type=int, default=None)
    parser.add_argument("--min-hand-coverage", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--start-after", type=float, default=None)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 = unlimited")
    parser.add_argument("--headless", action="store_true", help="No OpenCV window")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera image")
    parser.add_argument("--loop", action="store_true", help="Loop a video file")
    parser.add_argument(
        "--session-out",
        type=Path,
        default=None,
        help="Session JSON (default: reports/sentence_ollama_session_<time>.json)",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Offline debug: echo the glosses, no Ollama call",
    )
    parser.add_argument("--gap", type=float, default=None, help="Override sentence.gap_seconds")
    parser.add_argument(
        "--max-glosses", type=int, default=None, help="Override sentence.max_glosses"
    )
    parser.add_argument(
        "--max-wait", type=float, default=None, help="Override sentence.max_wait_seconds"
    )
    parser.add_argument(
        "--model", default=None, help="Override sentence.ollama_model"
    )
    parser.add_argument(
        "--url", default=None, help="Override sentence.ollama_url (Ollama server)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override sentence.ollama_timeout_seconds (cold model load)",
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=12.0,
        help="Seconds to wait for a pending rewrite when quitting (default: 12)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        settings = cfg.sentence

        gap_seconds = float(args.gap if args.gap is not None else settings.gap_seconds)
        max_glosses = int(
            args.max_glosses if args.max_glosses is not None else settings.max_glosses
        )
        max_wait = float(
            args.max_wait if args.max_wait is not None else settings.max_wait_seconds
        )
        ollama_model = args.model or settings.ollama_model
        ollama_url = args.url or settings.ollama_url
        timeout_seconds = float(
            args.timeout
            if args.timeout is not None
            else settings.ollama_timeout_seconds
        )
        if not ollama_url.strip():
            raise ConfigError("Ollama URL is empty; set sentence.ollama_url or --url.")

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

        if args.echo:
            rewriter = EchoRewriter()
        else:
            rewriter = OllamaRewriter(
                model=ollama_model,
                base_url=ollama_url,
                timeout_seconds=timeout_seconds,
                temperature=settings.temperature,
            )
        # Keep capture smooth even if a cold local model load takes a moment.
        runner = AsyncRewriter(rewriter)

        source = parse_source(
            args.source if args.source is not None else str(cfg.get("demo.camera", 0))
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
        utterances = UtteranceBuffer(
            gap_seconds=gap_seconds,
            max_glosses=max_glosses,
            max_wait_seconds=max_wait,
        )
        stats = DemoStats()
        records: list[dict] = []

        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_index = 0
        stream_time = 0.0
        started = time.perf_counter()
        last_ts = -1
        last_inference = -1e9
        last_sentence = ""
        status = "Sign one word at a time. Pause to finish the sentence."
        fps = 0.0
        last_frame_time = started

        def publish(
            utterance: Utterance, stream_time: float, result: RewriteResult
        ) -> None:
            """Record and display one finished rewrite."""
            nonlocal last_sentence, status
            if result.ok:
                last_sentence = result.text
                status = "Ollama rewrite" if not args.echo else "echo (offline)"
            elif result.warning:
                status = "Rewrite failed - see session log"
            records.append(
                {
                    **utterance.to_dict(),
                    "rewritten_at": round(float(stream_time), 3),
                    "backend": "echo" if args.echo else f"ollama:{ollama_model}",
                    "rewrite": result.to_dict(),
                }
            )
            if result.ok:
                print(f"              sentence: {result.text}")
                if result.dropped:
                    print(f"              dropped: {', '.join(result.dropped)}")
            elif result.warning:
                print(f"              warning: {result.warning}")

        def handle(utterance: Utterance | None, stream_time: float) -> None:
            """Print a completed utterance and queue it for rewriting."""
            nonlocal status
            if utterance is None:
                return
            print(
                f"\n[{stream_time:6.2f}s] glosses: {utterance.text or '(none)'} "
                f"({utterance.reason})"
            )
            if args.echo:
                status = "echo (offline)"
            else:
                status = "Ollama is rewriting..."
            runner.submit(utterance, stream_time)

        def drain() -> None:
            """Publish any rewrites finished since the last frame."""
            for utterance, stream_time, result in runner.poll():
                publish(utterance, stream_time, result)

        print(f"checkpoint : {checkpoint_path} (epoch {payload.get('epoch')})")
        print(f"source     : {source!r}")
        print(f"device     : {device}")
        print(
            f"gate       : window {window_seconds:.1f}s | threshold {threshold:.2f} | "
            f"streak {min_streak} | coverage {min_hand_coverage:.2f}"
        )
        print(
            f"sentence   : gap {gap_seconds:.1f}s | max {max_glosses} glosses / "
            f"{max_wait:.1f}s"
        )
        if args.echo:
            print("backend    : echo (offline debug)")
        else:
            print(f"backend    : {ollama_model} (local Ollama @ {ollama_url})")
            print(f"timeout    : {timeout_seconds:.0f}s (allows a cold model load)")
        if suppress_labels:
            print(f"suppressed : {', '.join(suppress_labels)} (never enters a sentence)")
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

                raw = detect_frame(landmarkers, image, timestamp, shoulder_visibility)
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

                    coverage = hand_coverage(buffer.frames())
                    decision = gate.update(
                        probs, coverage=coverage, timestamp=stream_time
                    )

                    # Feed the sentence buffer, not the overlay: rest is an
                    # explicit flush; suppressed commits are just no-sign time.
                    if decision.blocked or decision.index is None or decision.suppressed:
                        completed = utterances.on_no_sign(stream_time)
                    elif decision.index == rest_index:
                        completed = utterances.on_rest(stream_time)
                    else:
                        label = labels[decision.index]
                        completed = utterances.push(
                            label, cfg.display(label), stream_time
                        )
                    handle(completed, stream_time)
                    drain()

                    shown = (
                        "..."
                        if decision.index is None
                        else cfg.display(labels[decision.index])
                    )
                    stats.observe_prediction(
                        timestamp=stream_time,
                        label=(
                            None if decision.index is None else labels[decision.index]
                        ),
                        display=shown,
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
                    )
                    if args.headless:
                        print(
                            f"{stream_time:6.2f}s  {shown:<20} "
                            f"buffer: {utterances.text or '-':<40} "
                            f"hands {coverage:.2f}  fps {fps:5.1f}"
                        )

                if not args.headless:
                    gloss_text = utterances.text
                    cv2.putText(
                        frame,
                        "Glosses: " + (gloss_text or "(empty)"),
                        (18, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (220, 220, 220),
                        2,
                        cv2.LINE_AA,
                    )
                    framing = (
                        "hands out of frame - step back"
                        if wrist_out_of_frame(buffer.frames())
                        else status
                    )
                    cv2.putText(
                        frame,
                        framing,
                        (18, 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 180, 180),
                        2,
                        cv2.LINE_AA,
                    )
                    lines = wrap_caption(
                        _display_lines(gloss_text, last_sentence, ""),
                        font_scale=0.8,
                        thickness=2,
                        max_width=max(32, frame.shape[1] - 32),
                        max_lines=2,
                    )
                    draw_caption_bar(frame, lines, font_scale=0.8, thickness=2)
                    cv2.imshow("INCLUDE-50 sentence demo (experimental)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            handle(utterances.flush("eof", stream_time), stream_time)
            for utterance, finished_at, result in runner.close(
                timeout=max(0.0, float(args.shutdown_grace))
            ):
                publish(utterance, finished_at, result)
            if runner.pending:
                print(
                    f"Pending rewrites abandoned at exit: {runner.pending} "
                    "(the endpoint was still busy)"
                )
            cap.release()
            landmarkers.close()
            if not args.headless:
                cv2.destroyAllWindows()

        duration = time.perf_counter() - started
        summary = stats.summary(duration)
        print()
        print("=" * 70)
        print(
            f"Session: {duration:.1f}s, {summary['frames']} frames, "
            f"{summary['average_fps']} FPS, {summary['inferences']} inferences"
        )
        print(f"Utterances sent to the rewriter: {len(records)}")
        print(f"Last sentence: {last_sentence or '(none)'}")
        if utterances.dropped_oldest:
            print(f"Glosses dropped by the {max_glosses}-gloss cap: {utterances.dropped_oldest}")

        session_path = args.session_out or (
            cfg.path("evaluation.reports_dir")
            / f"sentence_ollama_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                        "infer_every_seconds": infer_every,
                        "gap_seconds": gap_seconds,
                        "max_glosses": max_glosses,
                        "max_wait_seconds": max_wait,
                        "ollama_model": ollama_model if not args.echo else "echo",
                        "ollama_url": ollama_url,
                        "temperature": settings.temperature,
                        "ollama_timeout_seconds": timeout_seconds,
                        "suppress_labels": suppress_labels,
                    },
                    "summary": summary,
                    "utterances": records,
                    "timeline": stats.timeline,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"Session log: {session_path}")
        print("Runs fully locally: no API key and no external network call.")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
