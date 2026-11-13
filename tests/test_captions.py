"""Caption transcript and overlay tests (no model, camera or video)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.captions import (  # noqa: E402
    PERIOD,
    CaptionState,
    draw_caption_bar,
    wrap_caption,
)


class TestCaptionState(unittest.TestCase):
    def test_committed_word_is_appended(self) -> None:
        captions = CaptionState()
        self.assertTrue(captions.push("cow", "Cow", 1.0))
        self.assertEqual(captions.words, ["Cow"])
        self.assertEqual(captions.text, "Cow")

    def test_uncommitted_decision_changes_nothing(self) -> None:
        captions = CaptionState()
        self.assertFalse(captions.push(None, "...", 1.0))
        self.assertEqual(captions.words, [])

    def test_rest_becomes_full_stop(self) -> None:
        captions = CaptionState()
        captions.push("cow", "Cow", 1.0)
        self.assertTrue(captions.push("rest", "No sign", 2.0, is_rest=True))
        self.assertEqual(captions.text, "Cow .")

    def test_no_leading_or_double_full_stop(self) -> None:
        captions = CaptionState()
        self.assertFalse(captions.push("rest", "No sign", 1.0, is_rest=True))
        captions.push("cow", "Cow", 2.0)
        captions.push("rest", "No sign", 3.0, is_rest=True)
        self.assertFalse(captions.push("rest", "No sign", 4.0, is_rest=True))
        self.assertEqual(captions.text, "Cow .")

    def test_held_label_does_not_stutter(self) -> None:
        captions = CaptionState()
        for t in (1.0, 1.33, 1.66, 2.0):
            captions.push("cow", "Cow", t)
        self.assertEqual(captions.words, ["Cow"])

    def test_word_can_repeat_after_a_full_stop(self) -> None:
        captions = CaptionState()
        captions.push("cow", "Cow", 1.0)
        captions.push("rest", "No sign", 2.0, is_rest=True)
        self.assertTrue(captions.push("cow", "Cow", 3.0))
        self.assertEqual(captions.text, "Cow . Cow")

    def test_max_words_drops_oldest(self) -> None:
        captions = CaptionState(max_words=3)
        for i, word in enumerate(("Cow", "Red", "Quiet", "Fan")):
            captions.push(word.lower(), word, float(i))
        self.assertEqual(captions.words, ["Red", "Quiet", "Fan"])

    def test_reset_clears_transcript(self) -> None:
        captions = CaptionState()
        captions.push("cow", "Cow", 1.0)
        captions.reset()
        self.assertEqual(captions.words, [])
        self.assertIsNone(captions.last_label)


class TestWrapCaption(unittest.TestCase):
    def test_empty_transcript(self) -> None:
        self.assertEqual(
            wrap_caption([], font_scale=0.8, thickness=2, max_width=200), []
        )

    def test_short_transcript_stays_on_one_line(self) -> None:
        lines = wrap_caption(
            ["Cow"], font_scale=0.8, thickness=2, max_width=400
        )
        self.assertEqual(lines, ["Cow"])

    def test_oldest_words_dropped_to_fit_two_lines(self) -> None:
        words = ["Cell phone", "Brother", "Quiet", "Red", "Cow"]
        lines = wrap_caption(
            words, font_scale=1.2, thickness=3, max_width=120, max_lines=2
        )
        self.assertLessEqual(len(lines), 2)
        self.assertIn("Cow", lines[-1])
        self.assertNotIn("Cell phone", " ".join(lines))

    def test_period_is_preserved(self) -> None:
        lines = wrap_caption(
            ["Cow", PERIOD, "Red"], font_scale=0.8, thickness=2, max_width=400
        )
        self.assertIn(PERIOD, " ".join(lines))


class TestDrawCaptionBar(unittest.TestCase):
    def test_bar_is_drawn_only_at_the_bottom(self) -> None:
        frame = np.full((240, 640, 3), 200, dtype=np.uint8)
        before = frame.copy()
        out = draw_caption_bar(frame, ["Cow . Red"], font_scale=0.8)

        self.assertTrue(np.array_equal(out[:60], before[:60]))
        self.assertFalse(np.array_equal(out[180:], before[180:]))

    def test_empty_lines_is_a_no_op(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        before = frame.copy()
        draw_caption_bar(frame, [])
        np.testing.assert_array_equal(frame, before)

    def test_tiny_frame_does_not_raise(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        draw_caption_bar(frame, ["Cow"], font_scale=0.8)
        self.assertEqual(frame.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
