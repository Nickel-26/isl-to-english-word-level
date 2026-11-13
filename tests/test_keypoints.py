"""Keypoint assembly tests (no videos or MediaPipe inference needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.keypoints import (  # noqa: E402
    FrameRaw,
    N_HAND,
    N_POSE,
    assemble_features,
    assign_hands,
    resample_sequence,
    timestamp_at,
)


def lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0, presence=1.0)


def pose_with_shoulders() -> list:
    pose = [lm(0.5, 0.5) for _ in range(N_POSE)]
    pose[11] = lm(0.4, 0.5)  # left shoulder
    pose[12] = lm(0.6, 0.5)  # right shoulder
    pose[0] = lm(0.5, 0.4)  # nose
    return pose


class TestAssembly(unittest.TestCase):
    def test_feature_dim_and_normalisation(self) -> None:
        hand = [lm(0.4, 0.5) for _ in range(N_HAND)]
        frame = FrameRaw(pose=pose_with_shoulders(), left_hand=hand, right_hand=None)
        features = assemble_features([frame])

        self.assertEqual(features.shape, (1, 137))

        # Nose: centre (0.5, 0.5), shoulder width 0.2 -> (0, -0.5).
        np.testing.assert_allclose(features[0, 0:2], [0.0, -0.5], atol=1e-6)

        # Left hand starts at the left wrist -> (-0.5, 0.0).
        offset = N_POSE * 2
        np.testing.assert_allclose(
            features[0, offset : offset + 2], [-0.5, 0.0], atol=1e-6
        )

        # Right hand absent -> zeros, and the presence flags are pose/left/right.
        right_offset = offset + N_HAND * 2
        self.assertFalse(features[0, right_offset : right_offset + 42].any())
        np.testing.assert_array_equal(features[0, -3:], [1.0, 1.0, 0.0])

    def test_resample_keeps_endpoints_and_rounds_flags(self) -> None:
        sequence = np.zeros((4, 137), dtype=np.float32)
        sequence[:, 0] = [0.0, 1.0, 2.0, 3.0]
        sequence[:, -3:] = np.tile([1, 0, 1], (4, 1)).astype(np.float32)

        out = resample_sequence(sequence, 8)

        self.assertEqual(out.shape, (8, 137))
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[-1, 0]), 3.0, places=5)
        self.assertTrue(set(np.unique(out[:, -3:])).issubset({0.0, 1.0}))

    def test_hand_assignment_uses_pose_wrists(self) -> None:
        pose = pose_with_shoulders()
        pose[15] = lm(0.35, 0.55)  # left wrist
        pose[16] = lm(0.65, 0.55)  # right wrist

        near_left = [lm(0.35, 0.55) for _ in range(N_HAND)]
        near_right = [lm(0.65, 0.55) for _ in range(N_HAND)]

        left, right = assign_hands([near_right, near_left], [], pose, 0.3)
        self.assertIs(left, near_left)
        self.assertIs(right, near_right)

        left, right = assign_hands([near_right], [], pose, 0.3)
        self.assertIsNone(left)
        self.assertIs(right, near_right)

    def test_single_frame_video_repeats(self) -> None:
        frame = FrameRaw(pose=pose_with_shoulders(), left_hand=None, right_hand=None)
        per_frame = assemble_features([frame])
        out = resample_sequence(per_frame, 32)
        self.assertEqual(out.shape, (32, 137))
        np.testing.assert_allclose(out[0], out[-1])

    def test_timestamps_stay_monotonic_across_clips(self) -> None:
        stamps: list[int] = []
        cursor = 0
        for n_frames in (5, 7, 3):
            last = cursor - 1
            for index in range(n_frames):
                ts = max(timestamp_at(cursor, index, 25.0), last + 1)
                self.assertGreater(ts, last)
                last = ts
                stamps.append(ts)
            cursor = last + 1

        self.assertEqual(stamps, sorted(set(stamps)))
        # The next clip must not reuse the previous clip's final timestamp.
        self.assertEqual(cursor, stamps[-1] + 1)


if __name__ == "__main__":
    unittest.main()
