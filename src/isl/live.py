"""Live-demo building blocks: timed frame buffer, windowing, probability smoothing.

Kept free of OpenCV so the logic can be unit-tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .keypoints import FrameRaw, assemble_features, resample_sequence


class TimedFrameBuffer:
    """Keep only the frames inside the last ``window_seconds``."""

    def __init__(self, window_seconds: float = 2.5):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = float(window_seconds)
        self._items: list[tuple[float, FrameRaw]] = []

    def append(self, timestamp: float, frame: FrameRaw) -> None:
        self._items.append((float(timestamp), frame))
        cutoff = timestamp - self.window_seconds
        self._items = [(t, f) for t, f in self._items if t >= cutoff]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def duration(self) -> float:
        if len(self._items) < 2:
            return 0.0
        return self._items[-1][0] - self._items[0][0]

    def frames(self) -> list[FrameRaw]:
        return [frame for _, frame in self._items]


def buffer_to_sequence(
    frames: list[FrameRaw],
    *,
    resample_frames: int = 32,
    pose_points: int = 25,
    use_z: bool = False,
    presence_flags: bool = True,
    shoulder_visibility: float = 0.3,
) -> np.ndarray:
    """Turn buffered raw frames into the same (32, 137) input used in training."""
    per_frame = assemble_features(
        frames,
        pose_points=pose_points,
        use_z=use_z,
        presence_flags=presence_flags,
        shoulder_visibility=shoulder_visibility,
    )
    return resample_sequence(
        per_frame, resample_frames, flags=3 if presence_flags else 0
    )


@dataclass
class SmoothedPrediction:
    index: int | None
    confidence: float
    raw_index: int
    raw_confidence: float
    streak: int


class ProbabilitySmoother:
    """EMA over class probabilities plus a minimum-streak gate.

    * ``alpha`` near 1 = trust the new frame, near 0 = heavy smoothing.
    * A class is only reported after ``min_streak`` consecutive smoothed
      argmaxes, and only if its smoothed probability clears ``threshold``.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        min_streak: int = 3,
        threshold: float = 0.6,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.min_streak = max(1, int(min_streak))
        self.threshold = float(threshold)
        self.ema: np.ndarray | None = None
        self.candidate: int | None = None
        self.streak = 0

    def reset(self) -> None:
        self.ema = None
        self.candidate = None
        self.streak = 0

    def update(self, probabilities: np.ndarray) -> SmoothedPrediction:
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")
        probs = probs / total

        if self.ema is None:
            self.ema = probs.copy()
        else:
            self.ema = self.alpha * probs + (1.0 - self.alpha) * self.ema

        raw_index = int(np.argmax(probs))
        smooth_index = int(np.argmax(self.ema))
        smooth_conf = float(self.ema[smooth_index])

        if smooth_index == self.candidate:
            self.streak += 1
        else:
            self.candidate = smooth_index
            self.streak = 1

        committed = (
            smooth_index
            if self.streak >= self.min_streak and smooth_conf >= self.threshold
            else None
        )
        return SmoothedPrediction(
            index=committed,
            confidence=smooth_conf,
            raw_index=raw_index,
            raw_confidence=float(probs[raw_index]),
            streak=self.streak,
        )


@dataclass
class DemoStats:
    """Counters for the end-of-session failure analysis."""

    frames: int = 0
    frames_without_hands: int = 0
    frames_without_pose: int = 0
    inferences: int = 0
    low_confidence: int = 0
    rest_predictions: int = 0
    label_switches: int = 0
    last_label: str | None = None
    label_counts: dict[str, int] = field(default_factory=dict)
    timeline: list[dict] = field(default_factory=list)

    def observe_frame(self, frame: FrameRaw) -> None:
        self.frames += 1
        if frame.pose is None:
            self.frames_without_pose += 1
        if frame.left_hand is None and frame.right_hand is None:
            self.frames_without_hands += 1

    def observe_prediction(
        self,
        *,
        timestamp: float,
        label: str | None,
        display: str,
        confidence: float,
        raw_label: str,
        raw_confidence: float,
        streak: int,
        fps: float,
        rest_label: str | None,
        threshold: float,
    ) -> None:
        self.inferences += 1
        if confidence < threshold:
            self.low_confidence += 1
        if label == rest_label:
            self.rest_predictions += 1
        if label != self.last_label:
            self.label_switches += 1
            self.last_label = label
        key = display if label is not None else "(below threshold)"
        self.label_counts[key] = self.label_counts.get(key, 0) + 1
        self.timeline.append(
            {
                "t": round(float(timestamp), 3),
                "label": label,
                "display": display,
                "confidence": round(float(confidence), 4),
                "raw_label": raw_label,
                "raw_confidence": round(float(raw_confidence), 4),
                "streak": int(streak),
                "fps": round(float(fps), 1),
            }
        )

    def summary(self, duration: float) -> dict:
        return {
            "duration_seconds": round(float(duration), 2),
            "frames": self.frames,
            "inferences": self.inferences,
            "average_fps": round(self.frames / duration, 1) if duration else 0.0,
            "frames_without_hands": self.frames_without_hands,
            "frames_without_pose": self.frames_without_pose,
            "low_confidence_inferences": self.low_confidence,
            "rest_predictions": self.rest_predictions,
            "label_switches": self.label_switches,
            "label_time": dict(
                sorted(self.label_counts.items(), key=lambda kv: -kv[1])
            ),
        }
