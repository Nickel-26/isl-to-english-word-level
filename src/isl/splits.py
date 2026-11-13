"""Reading and validating the official train/val/test split files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, ConfigError

VIDEO_SUFFIXES = (".mov", ".mp4", ".avi", ".mkv")


def normalise_rel_path(path: str) -> str:
    """Normalise a split/zip path to forward slashes and reject unsafe paths."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or p.startswith("~"):
        raise ConfigError(f"Unsafe split path (must be relative): {path!r}")
    if any(part == ".." for part in p.split("/")):
        raise ConfigError(f"Unsafe split path (contains '..'): {path!r}")
    return p


def folder_of(path: str) -> str:
    """``Places/23. Court/MVI_1.MOV`` -> ``Places/23. Court``."""
    parts = normalise_rel_path(path).split("/")
    if len(parts) < 2:
        raise ConfigError(f"Split path has no parent folder: {path!r}")
    return "/".join(parts[:-1])


def sign_folder_of(path: str) -> str:
    """Return ``Category/Sign`` even when the clip sits in an ``Extra/`` subfolder.

    Some official split entries look like
    ``Places/28. Store or Shop/Extra/MVI_3636.MOV``. The label always comes
    from the second path component, never from the immediate parent.
    """
    parts = normalise_rel_path(path).split("/")
    if len(parts) < 3:
        raise ConfigError(
            f"Expected <Category>/<NN. Sign>[/Extra]/<video>, got {path!r}"
        )
    return f"{parts[0]}/{parts[1]}"


def filename_of(path: str) -> str:
    return normalise_rel_path(path).split("/")[-1]


def video_key(path: str) -> str:
    """Extension-independent identity for a clip.

    The official split lists many clips as ``.MOV`` while the Zenodo ZIP
    contains the same recording as ``.MP4`` (and occasionally the reverse).
    ``MVI_8817.MOV`` and ``MVI_8817.MP4`` share the key
    ``home/40. paint/mvi_8817``, so matching by key tolerates that.
    """
    p = normalise_rel_path(path)
    folder, _, name = p.rpartition("/")
    stem = name.rsplit(".", 1)[0]
    return f"{folder}/{stem}".lower()


def index_videos_on_disk(raw_dir: Path) -> dict[str, Path]:
    """Map ``video_key`` -> actual file path for every video under ``raw_dir``."""
    out: dict[str, Path] = {}
    if not raw_dir.is_dir():
        return out
    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        rel = p.relative_to(raw_dir).as_posix()
        out[video_key(rel)] = p
    return out


def label_from_path(path: str, cfg: Config) -> str:
    """Read the sign label from the sign folder (second path component)."""
    parts = normalise_rel_path(path).split("/")
    if len(parts) < 3:
        raise ConfigError(
            f"Expected <Category>/<NN. Sign>[/Extra]/<video>, got {path!r}"
        )
    return cfg.normalise(parts[1])


def split_file_path(cfg: Config, split: str) -> Path:
    filename = cfg.get(f"dataset.split_files.{split}")
    return cfg.path("dataset.splits_dir") / str(filename)


def read_split_file(path: Path) -> list[str]:
    if not path.is_file():
        raise ConfigError(f"Split file not found: {path}")
    out: list[str] = []
    text = path.read_text(encoding="utf-8-sig")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(normalise_rel_path(line))
    return out


@dataclass
class SplitReport:
    splits: dict[str, list[str]]
    counts: dict[str, int]
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_splits(cfg: Config, strict: bool = True) -> SplitReport:
    """Load all split files and check labels, duplicates, overlaps and counts."""
    splits: dict[str, list[str]] = {}
    warnings: list[str] = []
    errors: list[str] = []

    allowed = set(cfg.labels)

    for split in ("train", "val", "test"):
        path = split_file_path(cfg, split)
        try:
            paths = read_split_file(path)
        except ConfigError as exc:
            if strict:
                raise
            errors.append(str(exc))
            splits[split] = []
            continue

        if not paths:
            errors.append(f"{split}: split file is empty ({path})")
        splits[split] = paths

        for p in paths:
            if not p.lower().endswith(VIDEO_SUFFIXES):
                errors.append(f"{split}: not a video path: {p}")
            try:
                label = label_from_path(p, cfg)
            except ConfigError as exc:
                errors.append(f"{split}: {exc}")
                continue
            if label not in allowed:
                errors.append(
                    f"{split}: label {label!r} ({p}) is not in dataset.signs"
                )

    counts = {s: len(v) for s, v in splits.items()}

    # Duplicates within a split (extension-insensitive, since ZIPs sometimes
    # store the same clip as .MP4 while the split says .MOV).
    for split, paths in splits.items():
        seen: set[str] = set()
        dupes: list[str] = []
        for p in paths:
            k = video_key(p)
            if k in seen:
                dupes.append(p)
            else:
                seen.add(k)
        if dupes:
            errors.append(
                f"{split}: {len(dupes)} duplicate clip(s), e.g. {sorted(set(dupes))[0]}"
            )

    # Overlap across splits.
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            keys_a = {video_key(p) for p in splits[a]}
            keys_b = {video_key(p) for p in splits[b]}
            overlap = keys_a & keys_b
            if overlap:
                errors.append(
                    f"{a}/{b} overlap on {len(overlap)} clip(s), e.g. {sorted(overlap)[0]}"
                )

    expected = cfg.get("dataset.split_source.expected_counts", {}) or {}
    for split, expected_n in expected.items():
        if split in counts and int(expected_n) != counts[split]:
            warnings.append(
                f"{split}: expected {expected_n} clips from the official split, "
                f"found {counts[split]}"
            )

    # Per-class counts.
    per_class: dict[str, dict[str, int]] = {}
    for split, paths in splits.items():
        for p in paths:
            try:
                label = label_from_path(p, cfg)
            except ConfigError:
                continue
            per_class.setdefault(label, {}).setdefault(split, 0)
            per_class[label][split] += 1

    missing_classes = sorted(allowed - set(per_class))
    if missing_classes:
        errors.append(
            f"{len(missing_classes)} sign(s) have no clips in any split: {missing_classes}"
        )

    return SplitReport(
        splits=splits,
        counts=counts,
        per_class=per_class,
        warnings=warnings,
        errors=errors,
    )


def all_listed_paths(report: SplitReport) -> set[str]:
    return {p for paths in report.splits.values() for p in paths}


def target_folders(report: SplitReport) -> set[str]:
    return {sign_folder_of(p) for p in all_listed_paths(report)}
