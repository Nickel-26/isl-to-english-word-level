"""Live-demo logic tests: timed buffer, windowing, smoothing (no camera)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.keypoints import FrameRaw, N_HAND, N_POSE  # noqa: E402
from isl.live import (  # noqa: E402
    ProbabilitySmoother,
    TimedFrameBuffer,
    buffer_to_sequence,
)


def lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0, presence=1.0)


def neutral_frame() -> FrameRaw:
    pose = [lm(0.5, 0.5) for _ in range(N_POSE)]
    pose[11] = lm(0.4, 0.5)
    pose[12] = lm(0.6, 0.5)
    return FrameRaw(
        pose=pose,
        left_hand=[lm(0.4, 0.5) for _ in range(N_HAND)],
        right_hand=[lm(0.6, 0.5) for _ in range(N_HAND)],
    )


class TestBuffer(unittest.TestCase):
    def test_window_drops_old_frames(self) -> None:
        buffer = TimedFrameBuffer(2.5)
        for step in range(60):
            buffer.append(step * 0.1, neutral_frame())
        self.assertLessEqual(buffer.duration, 2.5)
        self.assertGreaterEqual(len(buffer), 24)

    def test_empty_and_clear(self) -> None:
        buffer = TimedFrameBuffer(1.0)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.duration, 0.0)
        buffer.append(0.0, neutral_frame())
        buffer.clear()
        self.assertEqual(len(buffer), 0)

    def test_sequence_shape(self) -> None:
        frames = [neutral_frame() for _ in range(11)]
        sequence = buffer_to_sequence(frames, resample_frames=32)
        self.assertEqual(sequence.shape, (32, 137))
        self.assertTrue(np.isfinite(sequence).all())
        self.assertTrue(set(np.unique(sequence[:, -3:])).issubset({0.0, 1.0}))


class TestSmoother(unittest.TestCase):
    def test_needs_streak_and_threshold(self) -> None:
        smoother = ProbabilitySmoother(alpha=1.0, min_streak=3, threshold=0.6)
        probs = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        first = smoother.update(probs)
        second = smoother.update(probs)
        third = smoother.update(probs)
        self.assertIsNone(first.index)
        self.assertIsNone(second.index)
        self.assertEqual(third.index, 0)
        self.assertAlmostEqual(third.confidence, 0.7, places=5)
        self.assertEqual(third.streak, 3)

    def test_low_confidence_never_commits(self) -> None:
        smoother = ProbabilitySmoother(alpha=1.0, min_streak=1, threshold=0.8)
        for _ in range(10):
            out = smoother.update(np.array([0.5, 0.3, 0.2], dtype=np.float32))
            self.assertIsNone(out.index)

    def test_ema_smooths_a_spike(self) -> None:
        smoother = ProbabilitySmoother(alpha=0.5, min_streak=1, threshold=0.5)
        smoother.update(np.array([1.0, 0.0], dtype=np.float32))
        out = smoother.update(np.array([0.0, 1.0], dtype=np.float32))
        # EMA is halfway, so the smoothed score is not enough to switch yet.
        self.assertAlmostEqual(out.confidence, 0.5, places=5)
        self.assertEqual(out.raw_index, 1)
        out2 = smoother.update(np.array([0.0, 1.0], dtype=np.float32))
        self.assertEqual(out2.index, 1)

    def test_reset(self) -> None:
        smoother = ProbabilitySmoother(alpha=1.0, min_streak=1, threshold=0.5)
        smoother.update(np.array([1.0, 0.0], dtype=np.float32))
        smoother.reset()
        out = smoother.update(np.array([0.0, 1.0], dtype=np.float32))
        self.assertEqual(out.index, 1)
        self.assertEqual(out.streak, 1)

    def test_normalises_input(self) -> None:
        smoother = ProbabilitySmoother(alpha=1.0, min_streak=1, threshold=0.5)
        out = smoother.update(np.array([2.0, 2.0], dtype=np.float32))
        self.assertAlmostEqual(out.confidence, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
