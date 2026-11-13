"""Dataset, augmentation, rest synthesis and model shape tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.config import load_config  # noqa: E402
from isl.dataset import (  # noqa: E402
    FLAGS,
    HAND_POINTS,
    POSE_POINTS,
    augment_sequence,
    build_arrays,
    class_counts,
    mirror_sequence,
    synthesize_rest,
)
from isl.model import (  # noqa: E402
    BiGRUClassifier,
    TransformerClassifier,
    build_model,
    parameter_count,
)

DIM = POSE_POINTS * 2 + 2 * HAND_POINTS * 2 + FLAGS
CFG_PATH = ROOT / "configs" / "include50.yaml"


def fake_pool(n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    pool = rng.normal(0.0, 0.2, (n, 32, DIM)).astype(np.float32)
    pool[:, :, -FLAGS:] = 1.0
    # Put the hips somewhere sensible so rest hands have anchors.
    for i in range(POSE_POINTS):
        pool[:, :, i * 2] = 0.0
        pool[:, :, i * 2 + 1] = 0.0
    pool[:, :, 23 * 2] = -0.4
    pool[:, :, 24 * 2] = 0.4
    return pool


class TestAugment(unittest.TestCase):
    def test_shape_and_binary_flags(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            out = augment_sequence(
                fake_pool(1)[0],
                rng,
                enabled=True,
                p=1.0,
                time_shift=2,
                scale_range=(0.9, 1.15),
                speed_range=(0.8, 1.25),
            )
            self.assertEqual(out.shape, (32, DIM))
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue(set(np.unique(out[:, -FLAGS:])).issubset({0.0, 1.0}))

    def test_slow_speed_handles_padding(self) -> None:
        rng = np.random.default_rng(2)
        out = augment_sequence(
            fake_pool(1)[0],
            rng,
            enabled=True,
            p=1.0,
            time_shift=0,
            scale_range=(1.0, 1.0),
            speed_range=(0.8, 0.8),
        )
        self.assertEqual(out.shape, (32, DIM))
        self.assertTrue(np.isfinite(out).all())

    def test_scale_changes_coordinates_not_flags(self) -> None:
        base = fake_pool(1)[0]
        out = augment_sequence(
            base,
            np.random.default_rng(3),
            enabled=True,
            p=1.0,
            time_shift=0,
            scale_range=(1.1, 1.1),
            speed_range=(1.0, 1.0),
        )
        np.testing.assert_allclose(out[:, :-FLAGS], base[:, :-FLAGS] * 1.1, atol=1e-5)
        np.testing.assert_array_equal(out[:, -FLAGS:], base[:, -FLAGS:])

    def test_mirror_negates_x_and_swaps_hands(self) -> None:
        base = fake_pool(1)[0].copy()
        # Distinct values so a mistaken swap or sign flip cannot pass by luck.
        base[:, 0 : POSE_POINTS * 2 : 2] = np.arange(POSE_POINTS)[None, :]
        base[:, 1 : POSE_POINTS * 2 : 2] = 100.0
        base[:, POSE_POINTS * 2 : (POSE_POINTS + HAND_POINTS) * 2] = 7.0
        base[:, (POSE_POINTS + HAND_POINTS) * 2 : (POSE_POINTS + 2 * HAND_POINTS) * 2] = 9.0
        base[:, -2] = 1.0
        base[:, -1] = 0.0

        out = mirror_sequence(base)
        np.testing.assert_allclose(
            out[:, 0 : POSE_POINTS * 2 : 2], -base[:, 0 : POSE_POINTS * 2 : 2]
        )
        left = out[:, POSE_POINTS * 2 : (POSE_POINTS + HAND_POINTS) * 2]
        right = out[:, (POSE_POINTS + HAND_POINTS) * 2 : (POSE_POINTS + 2 * HAND_POINTS) * 2]
        np.testing.assert_allclose(left[:, 0::2], -9.0)
        np.testing.assert_allclose(left[:, 1::2], 9.0)
        np.testing.assert_allclose(right[:, 0::2], -7.0)
        np.testing.assert_allclose(right[:, 1::2], 7.0)
        self.assertEqual(out[0, -2], 0.0)
        self.assertEqual(out[0, -1], 1.0)
        # Input must not be mutated.
        self.assertEqual(base[0, -2], 1.0)

    def test_mirror_one_handed_flips_only_single_hand_sequences(self) -> None:
        base = fake_pool(1)[0].copy()
        base[:, 0 : POSE_POINTS * 2 : 2] = np.arange(POSE_POINTS)[None, :]
        base[:, -2] = 0.0
        base[:, -1] = 1.0

        out = augment_sequence(
            base,
            np.random.default_rng(7),
            enabled=True,
            p=0.0,
            mirror_one_handed=1.0,
        )
        # x negated and hands swapped: the right hand becomes the left hand.
        np.testing.assert_allclose(
            out[:, 0 : POSE_POINTS * 2 : 2], -base[:, 0 : POSE_POINTS * 2 : 2]
        )
        self.assertEqual(float(out[:, -2].mean()), 1.0)
        self.assertEqual(float(out[:, -1].mean()), 0.0)

        # Two-handed sequences must be left untouched.
        two_handed = fake_pool(1)[0].copy()
        two_handed[:, -2] = 1.0
        two_handed[:, -1] = 1.0
        unchanged = augment_sequence(
            two_handed,
            np.random.default_rng(8),
            enabled=True,
            p=0.0,
            mirror_one_handed=1.0,
        )
        np.testing.assert_array_equal(unchanged, two_handed)

    def test_hand_dropout_masks_hand_and_flag_together(self) -> None:
        base = fake_pool(1)[0]
        out = augment_sequence(
            base,
            np.random.default_rng(5),
            enabled=True,
            p=0.0,
            hand_dropout=1.0,
            hand_dropout_span=6,
        )
        for flag, block in (
            (-2, slice(POSE_POINTS * 2, (POSE_POINTS + HAND_POINTS) * 2)),
            (-1, slice((POSE_POINTS + HAND_POINTS) * 2, (POSE_POINTS + 2 * HAND_POINTS) * 2)),
        ):
            missing = out[:, flag] == 0.0
            self.assertTrue(missing.any())
            self.assertTrue((out[missing][:, block] == 0.0).all())
            self.assertTrue((out[~missing][:, flag] == 1.0).all())

    def test_hand_dropout_off_is_identity(self) -> None:
        base = fake_pool(1)[0]
        out = augment_sequence(
            base, np.random.default_rng(6), enabled=True, p=0.0, hand_dropout=0.0
        )
        np.testing.assert_array_equal(out, base)


class TestRest(unittest.TestCase):
    def test_rest_sequences_are_finite_and_neutral(self) -> None:
        pool = fake_pool(3)
        rest = synthesize_rest(pool, 20, rng=np.random.default_rng(4))
        self.assertEqual(rest.shape, (20, 32, DIM))
        self.assertTrue(np.isfinite(rest).all())
        self.assertTrue(set(np.unique(rest[:, :, -FLAGS:])).issubset({0.0, 1.0}))
        self.assertTrue((rest[:, :, -3] == 1.0).all())
        # Hand presence must vary across examples.
        self.assertGreater(len(set(np.unique(rest[:, 0, -2:]))), 1)

    def test_count_zero(self) -> None:
        rest = synthesize_rest(fake_pool(1), 0, rng=np.random.default_rng(0))
        self.assertEqual(rest.shape[0], 0)


class TestModel(unittest.TestCase):
    def test_bigru_forward(self) -> None:
        model = BiGRUClassifier(input_dim=DIM, num_classes=51, hidden_size=32)
        out = model(torch.zeros((2, 32, DIM), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (2, 51))

    def test_transformer_forward(self) -> None:
        model = TransformerClassifier(
            input_dim=DIM, num_classes=51, seq_len=32, hidden_size=32, nhead=4
        )
        out = model(torch.zeros((2, 32, DIM), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (2, 51))

    def test_build_model_from_config(self) -> None:
        if not CFG_PATH.is_file():
            self.skipTest("config not available")
        cfg = load_config(CFG_PATH)
        model = build_model(cfg)
        out = model(torch.zeros((1, 32, cfg.feature_dim), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (1, cfg.num_classes))
        self.assertGreater(parameter_count(model), 0)


class TestCacheIntegration(unittest.TestCase):
    def test_load_real_cache_if_present(self) -> None:
        if not CFG_PATH.is_file():
            self.skipTest("config not available")
        cfg = load_config(CFG_PATH)
        if not cfg.path("keypoints.cache_dir").is_dir():
            self.skipTest("keypoint cache not extracted yet")

        arrays = build_arrays(cfg)
        train_x, train_y = arrays["train"]
        val_x, val_y = arrays["val"]
        test_x, test_y = arrays["test"]

        self.assertEqual(test_x.shape, (192, 32, cfg.feature_dim))
        self.assertEqual(len(test_x), len(test_y))
        self.assertEqual(
            train_x.shape[0], 689 + int(cfg.get("dataset.rest.synthetic_examples"))
        )
        self.assertEqual(
            val_x.shape[0], 77 + int(cfg.get("dataset.rest.val_examples"))
        )
        counts = class_counts(train_y, cfg.num_classes)
        self.assertEqual(int(counts[-1]), int(cfg.get("dataset.rest.synthetic_examples")))
        self.assertTrue(np.isfinite(train_x).all())


if __name__ == "__main__":
    unittest.main()
