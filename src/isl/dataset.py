"""Training data: keypoint cache loading, augmentation, synthetic rest class.

The cache is one ``.npz`` per clip, written by ``scripts/extract_keypoints.py``.
Sequences are ``(32, 137)`` float32:

    pose 25 points x/y  -> 0:50
    left hand 21 x/y    -> 50:92
    right hand 21 x/y   -> 92:134
    presence flags      -> 134:137  (pose, left hand, right hand)

All coordinates are already shoulder-relative (centre = shoulder midpoint,
scale = shoulder width), so augmentation only needs small shifts, scale changes
and speed changes.

There are no real "no sign" clips in INCLUDE, so the rest class is synthesised
from real poses with the hands placed at the hips. This teaches the model what
a neutral standing pose looks like; the live demo additionally gates on
confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config, ConfigError
from .splits import label_from_path, load_splits

POSE_POINTS = 25
HAND_POINTS = 21
FLAGS = 3
L_HIP = 23
R_HIP = 24


def cache_path(cfg: Config, label: str, source: str) -> Path:
    """Cache file for a split entry (must match the extractor's naming)."""
    return cfg.path("keypoints.cache_dir") / label / f"{Path(source).stem}.npz"


def split_sources(cfg: Config, split: str, report=None) -> list[str]:
    if report is None:
        report = load_splits(cfg, strict=True)
    try:
        return list(report.splits[split])
    except KeyError as exc:
        raise ConfigError(f"Unknown split: {split!r}") from exc


def load_split(
    cfg: Config, split: str, report=None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one split from the keypoint cache.

    Returns ``(X[N, T, D], y[N], sources)`` where ``y`` uses the class order in
    the config (signs first, rest last).
    """
    sources = split_sources(cfg, split, report)
    seq_len = int(cfg.get("keypoints.resample_frames", 32))
    dim = int(cfg.feature_dim)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    kept: list[str] = []

    for source in sources:
        label = label_from_path(source, cfg)
        if label not in cfg.label_to_idx:
            raise ConfigError(f"{split}: label {label!r} is not in dataset.signs")
        path = cache_path(cfg, label, source)
        if not path.is_file():
            raise ConfigError(
                f"Missing keypoint cache for {source}\n"
                f"  expected: {path}\n"
                f"  run: python scripts/extract_keypoints.py"
            )
        with np.load(path, allow_pickle=False) as data:
            features = np.asarray(data["X"], dtype=np.float32)
            stored = str(data["label"]) if "label" in data else label
        if stored != label:
            raise ConfigError(
                f"Cache label mismatch for {path}: {stored!r} != {label!r}"
            )
        if features.shape != (seq_len, dim):
            raise ConfigError(
                f"Cache shape {features.shape} in {path}, expected {(seq_len, dim)}. "
                "Re-run extraction after changing keypoint settings."
            )
        xs.append(features)
        ys.append(cfg.label_to_idx[label])
        kept.append(source)

    if not xs:
        raise ConfigError(f"Split {split!r} has no clips.")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), kept


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def _interp_rows(sequence: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate every column at fractional source positions (edge-clamped)."""
    time = np.arange(sequence.shape[0], dtype=np.float64)
    out = np.empty((len(positions), sequence.shape[1]), dtype=np.float32)
    for column in range(sequence.shape[1]):
        out[:, column] = np.interp(
            positions, time, sequence[:, column]
        ).astype(np.float32)
    return out


def _time_shift(sequence: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0:
        return sequence
    shift = int(rng.integers(-limit, limit + 1))
    if shift == 0:
        return sequence
    if shift > 0:
        pad = np.repeat(sequence[:1], shift, axis=0)
        return np.concatenate([pad, sequence[:-shift]], axis=0)
    pad = np.repeat(sequence[-1:], -shift, axis=0)
    return np.concatenate([sequence[-shift:], pad], axis=0)


def _speed(sequence: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Change motion speed while keeping 32 frames.

    Faster (factor > 1) samples a shorter window of the clip; slower pads the
    ends by repeating the first/last frame before sampling. The window is
    interpolated with all columns, so the presence flags are rounded afterwards.
    """
    frames = sequence.shape[0]
    window = frames / factor
    if window <= frames:
        span = max(2.0, window)
        start = float(rng.uniform(0.0, max(0.0, frames - span)))
        positions = start + np.linspace(0.0, span - 1.0, frames)
        source = sequence
    else:
        pad_total = int(np.ceil(window - frames))
        source = np.concatenate(
            [
                np.repeat(sequence[:1], pad_total, axis=0),
                sequence,
                np.repeat(sequence[-1:], pad_total, axis=0),
            ],
            axis=0,
        )
        start = float(rng.uniform(0.0, max(0.0, len(source) - window)))
        positions = start + np.linspace(0.0, window - 1.0, frames)

    out = _interp_rows(source, positions)
    out[:, -FLAGS:] = (out[:, -FLAGS:] >= 0.5).astype(np.float32)
    return out


def augment_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator,
    *,
    enabled: bool = True,
    p: float = 0.6,
    time_shift: int = 2,
    scale_range: tuple[float, float] = (0.9, 1.15),
    speed_range: tuple[float, float] = (0.8, 1.25),
) -> np.ndarray:
    """Apply random shift/scale/speed, leaving presence flags as 0/1."""
    out = np.array(sequence, dtype=np.float32, copy=True)
    if not enabled:
        return out

    if rng.random() < p:
        out = _time_shift(out, int(time_shift), rng)

    if rng.random() < p:
        scale = float(rng.uniform(*scale_range))
        out[:, :-FLAGS] *= scale

    if rng.random() < p:
        out = _speed(out, float(rng.uniform(*speed_range)), rng)

    out[:, -FLAGS:] = (out[:, -FLAGS:] >= 0.5).astype(np.float32)
    return out


def augment_from_config(
    sequence: np.ndarray, rng: np.random.Generator, cfg: Config, enabled: bool = True
) -> np.ndarray:
    scale = cfg.get("augment.scale", [0.9, 1.15])
    speed = cfg.get("augment.speed", [0.8, 1.25])
    return augment_sequence(
        sequence,
        rng,
        enabled=enabled and bool(cfg.get("augment.enabled", True)),
        p=float(cfg.get("augment.p", 0.6)),
        time_shift=int(cfg.get("augment.time_shift", 2)),
        scale_range=(float(scale[0]), float(scale[1])),
        speed_range=(float(speed[0]), float(speed[1])),
    )


def synthesize_rest(
    pool: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    hand_present_prob: float = 0.7,
) -> np.ndarray:
    """Build neutral "no sign" sequences from real poses.

    The pose is the median pose of a real clip (so the skeleton is realistic),
    with a slow sway. Hands sit at the hips. Hands are marked absent for a
    fraction of examples, mirroring frames where MediaPipe misses them.
    """
    if count <= 0:
        return np.zeros((0, pool.shape[1], pool.shape[2]), dtype=np.float32)
    if pool.shape[2] != POSE_POINTS * 2 + 2 * HAND_POINTS * 2 + FLAGS:
        raise ConfigError(
            "Rest synthesis expects x/y keypoints at the default layout "
            f"({POSE_POINTS * 2 + 2 * HAND_POINTS * 2 + FLAGS} dims), got {pool.shape[2]}."
        )

    frames = pool.shape[1]
    out = np.zeros((count, frames, pool.shape[2]), dtype=np.float32)
    time = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)

    for i in range(count):
        base = pool[int(rng.integers(0, len(pool)))]
        pose = np.median(base[:, : POSE_POINTS * 2], axis=0).reshape(POSE_POINTS, 2)

        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        sway = 0.015 * np.sin(time + phase)[:, None, None]
        jitter = rng.normal(0.0, 0.004, (frames, POSE_POINTS, 2)).astype(np.float32)
        pose_frames = pose[None, :, :] + sway + jitter

        left_hip = pose[L_HIP]
        right_hip = pose[R_HIP]
        hand_noise_l = rng.normal(0.0, 0.015, (frames, HAND_POINTS, 2)).astype(np.float32)
        hand_noise_r = rng.normal(0.0, 0.015, (frames, HAND_POINTS, 2)).astype(np.float32)
        left = left_hip[None, None, :] + hand_noise_l
        right = right_hip[None, None, :] + hand_noise_r

        left_present = rng.random() < hand_present_prob
        right_present = rng.random() < hand_present_prob
        out[i, :, : POSE_POINTS * 2] = pose_frames.reshape(frames, -1)
        if left_present:
            out[i, :, POSE_POINTS * 2 : (POSE_POINTS + HAND_POINTS) * 2] = left.reshape(
                frames, -1
            )
        if right_present:
            offset = (POSE_POINTS + HAND_POINTS) * 2
            out[i, :, offset : offset + HAND_POINTS * 2] = right.reshape(frames, -1)
        out[i, :, -FLAGS:] = (1.0, float(left_present), float(right_present))

    return out


def build_arrays(
    cfg: Config, report=None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all splits and add synthetic rest to train/val.

    Test stays exactly the official 192 clips so evaluation numbers remain
    comparable with the published split. Rest false positives are assessed
    separately in the demo/evaluation stage.
    """
    if bool(cfg.get("keypoints.use_z", False)):
        raise ConfigError("Rest synthesis assumes x/y keypoints (keypoints.use_z: false).")

    splits = {name: load_split(cfg, name, report) for name in ("train", "val", "test")}
    rng = np.random.default_rng(int(cfg.get("project.seed", 42)))

    train_x, train_y, _ = splits["train"]
    val_x, val_y, _ = splits["val"]

    if cfg.rest_enabled:
        rest_idx = cfg.label_to_idx[cfg.rest_label]
        train_rest_n = int(cfg.get("dataset.rest.synthetic_examples", 0))
        val_rest_n = int(cfg.get("dataset.rest.val_examples", 0))
        if train_rest_n > 0:
            train_x = np.concatenate(
                [train_x, synthesize_rest(train_x, train_rest_n, rng=rng)], axis=0
            )
            train_y = np.concatenate(
                [train_y, np.full(train_rest_n, rest_idx, dtype=np.int64)], axis=0
            )
        if val_rest_n > 0:
            val_x = np.concatenate(
                [val_x, synthesize_rest(val_x, val_rest_n, rng=rng)], axis=0
            )
            val_y = np.concatenate(
                [val_y, np.full(val_rest_n, rest_idx, dtype=np.int64)], axis=0
            )

    return {
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "test": (splits["test"][0], splits["test"][1]),
    }


class KeypointDataset:
    """Minimal torch Dataset (imported lazily so numpy-only tools can use this file)."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        cfg: Config | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        self.features = features
        self.labels = labels
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        import torch

        sequence = self.features[index]
        if self.augment and self.cfg is not None:
            sequence = augment_from_config(sequence, self.rng, self.cfg, enabled=True)
        return torch.from_numpy(np.ascontiguousarray(sequence)), int(self.labels[index])
