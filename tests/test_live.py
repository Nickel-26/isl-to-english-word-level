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
    DecisionGate,
    DemoStats,
    ProbabilitySmoother,
    TimedFrameBuffer,
    buffer_to_sequence,
    hand_coverage,
    wrist_out_of_frame,
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


class TestEvidence(unittest.TestCase):
    def test_hand_coverage_counts_either_hand(self) -> None:
        frames = [neutral_frame(), neutral_frame(), neutral_frame()]
        frames[0].left_hand = None
        frames[1].left_hand = None
        frames[1].right_hand = None
        frames[2].right_hand = None
        self.assertAlmostEqual(hand_coverage(frames), 2 / 3, places=6)
        self.assertEqual(hand_coverage([]), 0.0)

    def test_wrist_out_of_frame_detects_extrapolation(self) -> None:
        frame = neutral_frame()
        self.assertFalse(wrist_out_of_frame([frame]))

        frame.pose[15] = lm(1.35, 0.5)
        self.assertTrue(wrist_out_of_frame([frame]))

    def test_wrist_out_of_frame_without_pose(self) -> None:
        frame = neutral_frame()
        frame.pose = None
        self.assertTrue(wrist_out_of_frame([frame]))


class TestDecisionGate(unittest.TestCase):
    def test_low_coverage_blocks_and_does_not_commit(self) -> None:
        gate = DecisionGate(
            alpha=1.0, min_streak=1, threshold=0.5, min_hand_coverage=0.3
        )
        decision = gate.update(
            np.array([0.9, 0.1], dtype=np.float32), coverage=0.1, timestamp=1.0
        )
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "low_hand_coverage")
        self.assertIsNone(decision.index)
        self.assertEqual(gate.blocked_count, 1)

    def test_coverage_gate_keeps_streak_within_reset_window(self) -> None:
        gate = DecisionGate(
            alpha=1.0,
            min_streak=3,
            threshold=0.5,
            min_hand_coverage=0.3,
            reset_after_missing_seconds=0.8,
        )
        probs = np.array([0.9, 0.1], dtype=np.float32)
        gate.update(probs, coverage=1.0, timestamp=0.0)
        gate.update(probs, coverage=1.0, timestamp=0.3)
        blocked = gate.update(probs, coverage=0.0, timestamp=0.4)
        self.assertTrue(blocked.blocked)
        # Only 0.1 s without evidence: the EMA/streak must survive.
        decision = gate.update(probs, coverage=1.0, timestamp=0.5)
        self.assertFalse(decision.blocked)
        self.assertEqual(decision.index, 0)
        self.assertEqual(decision.streak, 3)

    def test_long_missing_resets_the_smoother(self) -> None:
        gate = DecisionGate(
            alpha=1.0,
            min_streak=2,
            threshold=0.5,
            min_hand_coverage=0.3,
            reset_after_missing_seconds=0.8,
        )
        probs = np.array([0.9, 0.1], dtype=np.float32)
        gate.update(probs, coverage=1.0, timestamp=0.0)
        gate.update(probs, coverage=0.0, timestamp=1.0)
        decision = gate.update(probs, coverage=1.0, timestamp=1.1)
        # Streak restarted, so a 2-streak gate is not satisfied yet.
        self.assertEqual(decision.streak, 1)
        self.assertIsNone(decision.index)


class TestSuppression(unittest.TestCase):
    def test_suppressed_commit_returns_no_index(self) -> None:
        gate = DecisionGate(
            alpha=1.0,
            min_streak=1,
            threshold=0.5,
            suppress_indices={1},
        )
        decision = gate.update(
            np.array([0.1, 0.8, 0.1], dtype=np.float32),
            coverage=1.0,
            timestamp=1.0,
        )
        self.assertIsNone(decision.index)
        self.assertTrue(decision.suppressed)
        self.assertFalse(decision.blocked)
        self.assertEqual(decision.reason, "suppressed_label")
        self.assertAlmostEqual(decision.confidence, 0.8, places=5)
        self.assertEqual(decision.streak, 1)
        self.assertEqual(gate.suppressed_count, 1)
        self.assertEqual(gate.blocked_count, 0)

    def test_non_suppressed_label_still_commits(self) -> None:
        gate = DecisionGate(
            alpha=1.0,
            min_streak=1,
            threshold=0.5,
            suppress_indices={2},
        )
        decision = gate.update(
            np.array([0.1, 0.8, 0.1], dtype=np.float32),
            coverage=1.0,
            timestamp=1.0,
        )
        self.assertEqual(decision.index, 1)
        self.assertFalse(decision.suppressed)
        self.assertEqual(decision.reason, "")

    def test_suppression_keeps_smoother_and_streak_running(self) -> None:
        gate = DecisionGate(
            alpha=1.0,
            min_streak=2,
            threshold=0.5,
            suppress_indices={0},
        )
        probs = np.array([0.9, 0.1], dtype=np.float32)
        first = gate.update(probs, coverage=1.0, timestamp=0.0)
        second = gate.update(probs, coverage=1.0, timestamp=0.3)
        self.assertIsNone(first.index)
        self.assertFalse(first.suppressed)
        self.assertIsNone(second.index)
        self.assertTrue(second.suppressed)
        # The streak grew even though nothing was displayed.
        self.assertEqual(second.streak, 2)
        self.assertAlmostEqual(second.confidence, 0.9, places=5)


class TestDemoStatsSuppression(unittest.TestCase):
    def test_suppressed_inference_is_counted_separately(self) -> None:
        stats = DemoStats()
        stats.observe_prediction(
            timestamp=1.0,
            label=None,
            display="...",
            confidence=0.9,
            raw_label="it",
            raw_confidence=0.9,
            streak=2,
            fps=20.0,
            rest_label="rest",
            threshold=0.6,
            hand_coverage=1.0,
            suppressed=True,
            reason="suppressed_label",
        )
        summary = stats.summary(1.0)
        self.assertEqual(summary["suppressed_inferences"], 1)
        self.assertEqual(summary["blocked_inferences"], 0)
        self.assertEqual(summary["label_time"], {"(suppressed)": 1})
        self.assertTrue(stats.timeline[0]["suppressed"])
        self.assertEqual(stats.timeline[0]["raw_label"], "it")


if __name__ == "__main__":
    unittest.main()
