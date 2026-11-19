"""Caption state and overlay drawing for recorded-video demos.

The transcript state machine is pure Python so it can be unit-tested without
OpenCV or a model. Only :func:`draw_caption_bar` touches pixels.

The demo semantics are deliberately conservative: a word enters the transcript
only when the decision gate *commits* it. Low-confidence windows and windows
with too few detected hands add nothing. The synthetic ``rest`` class becomes a
full stop, so a caption line reads like ``Cow . Red .``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

PERIOD = "."


@dataclass
class CaptionState:
    """Rolling transcript built from committed labels.

    Consecutive commits of the same label are collapsed: an isolated sign can
    stay on screen for several inference windows, and repeating it on every
    window would turn one word into a stutter. A word can appear again once a
    different label (including a full stop) has been committed in between.
    """

    max_words: int = 12
    words: list[str] = field(default_factory=list)
    last_label: str | None = None

    def push(
        self,
        label: str | None,
        display: str,
        timestamp: float,
        *,
        is_rest: bool = False,
    ) -> bool:
        """Record one gated decision. Returns True if the transcript changed."""
        if label is None:
            return False
        if label == self.last_label:
            return False

        word = PERIOD if is_rest else str(display).strip()
        if not word:
            return False

        self.last_label = label
        if word == PERIOD:
            # Never start a transcript with a full stop, and never double one.
            if not self.words or self.words[-1] == PERIOD:
                return False
        self.words.append(word)
        if len(self.words) > max(1, int(self.max_words)):
            del self.words[: len(self.words) - int(self.max_words)]
        return True

    def reset(self) -> None:
        self.words.clear()
        self.last_label = None

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def __len__(self) -> int:
        return len(self.words)


def sentence_at(entries: Sequence[tuple[float, str]], t: float) -> str:
    """Return the final sentence active at ``t``; entries are ``(start, text)``.

    Entries must be sorted by start time. An empty text deliberately clears the
    caption (for example when an utterance failed to rewrite). This is pure
    timing logic so the sentence-caption renderer can be tested without a video.
    """
    active = ""
    now = float(t)
    for start, text in entries:
        if float(start) > now:
            break
        active = str(text)
    return active


def _text_size(text: str, font_scale: float, thickness: int) -> tuple[int, int, int]:
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    return width, height, baseline


def _greedy_wrap(
    words: list[str], *, font_scale: float, thickness: int, max_width: int
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _text_size(candidate, font_scale, thickness)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def wrap_caption(
    words: list[str],
    *,
    font_scale: float,
    thickness: int,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    """Wrap a rolling transcript to fit, dropping the oldest words first."""
    tail = [str(w) for w in words]
    if not tail:
        return []
    while True:
        lines = _greedy_wrap(
            tail, font_scale=font_scale, thickness=thickness, max_width=max_width
        )
        too_wide = any(
            _text_size(line, font_scale, thickness)[0] > max_width for line in lines
        )
        if not too_wide and len(lines) <= max_lines:
            return lines
        if len(tail) <= 1:
            # A single unbreakable word: keep it; draw_caption_bar clips it.
            return lines[-max_lines:]
        tail.pop(0)


def draw_caption_bar(
    frame: np.ndarray,
    lines: list[str],
    *,
    font_scale: float | None = None,
    thickness: int | None = None,
    opacity: float = 0.55,
    text_colour: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw right-aligned caption lines in a translucent bar at the bottom."""
    if not lines or frame.size == 0:
        return frame

    height, width = frame.shape[:2]
    if font_scale is None:
        font_scale = max(0.45, height / 720.0 * 0.85)
    font_scale = float(font_scale)
    if thickness is None:
        thickness = max(1, int(round(font_scale * 2)))

    pad_x = max(8, int(round(width * 0.02)))
    pad_y = max(6, int(round(height * 0.018)))
    gap = max(4, int(round(font_scale * 10)))
    sizes = [_text_size(line, font_scale, thickness) for line in lines]
    line_heights = [text_h + base for text_w, text_h, base in sizes]
    block_height = sum(line_heights) + gap * (len(lines) - 1) + 2 * pad_y
    block_height = min(block_height, height)
    top = height - block_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (width, height), (0, 0, 0), -1)
    alpha = min(max(float(opacity), 0.0), 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)

    cursor = top + pad_y
    for line, (text_w, text_h, base), line_h in zip(lines, sizes, line_heights):
        x = max(pad_x, width - pad_x - text_w)
        baseline = min(height - pad_y, cursor + text_h)
        # Soft shadow keeps white text readable over bright backgrounds.
        shadow = max(1, thickness // 2)
        cv2.putText(
            frame,
            line,
            (x + shadow, baseline + shadow),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + shadow,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_colour,
            thickness,
            cv2.LINE_AA,
        )
        cursor += line_h + gap
    return frame
