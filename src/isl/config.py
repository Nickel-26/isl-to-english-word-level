"""Config loading and validation.

The YAML file is the single source of truth for the sign list. Everything
downstream (class count, label order, model output size) is derived here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


class ConfigError(ValueError):
    """Raised when the YAML config is missing or inconsistent."""


@dataclass(frozen=True)
class Sign:
    label: str
    display: str


def normalise_label(text: str, rule: str = "strip_nonalpha_lower") -> str:
    """Turn a folder like ``"54. Cell phone"`` into ``"cellphone"``."""
    if rule == "strip_nonalpha_lower":
        return "".join(ch for ch in text if ch.isalpha()).lower()
    raise ConfigError(f"Unknown dataset.label_rule: {rule!r}")


class Config:
    """Validated view over ``configs/include50.yaml``.

    Relative paths are resolved against the project root, which is inferred as
    the parent of the ``configs/`` directory unless ``root`` is given.
    """

    def __init__(self, path: str | Path, root: str | Path | None = None):
        self.config_path = Path(path).expanduser().resolve()
        if not self.config_path.is_file():
            raise ConfigError(f"Config file not found: {self.config_path}")

        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else self.config_path.parent.parent
        )

        with self.config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{self.config_path} must contain a YAML mapping at the top level.")
        self.raw: dict[str, Any] = raw

        self._validate()
        self._build_labels()

    # ------------------------------------------------------------------ public
    @property
    def project_name(self) -> str:
        return str(self.get("project.name"))

    @property
    def signs(self) -> list[Sign]:
        return list(self._signs)

    @property
    def labels(self) -> list[str]:
        """Sign labels in class-index order (rest excluded)."""
        return [s.label for s in self._signs]

    @property
    def displays(self) -> dict[str, str]:
        out = {s.label: s.display for s in self._signs}
        if self.rest_enabled:
            out[self.rest_label] = str(self.get("dataset.rest.display"))
        return out

    @property
    def rest_enabled(self) -> bool:
        return bool(self.get("dataset.rest.enabled", False))

    @property
    def rest_label(self) -> str:
        return "rest"

    @property
    def suppress_labels(self) -> list[str]:
        """Labels the demo hides even when the model commits them.

        This is a presentation-only setting: the checkpoint still predicts the
        label and official test metrics are unchanged.
        """
        value = self.get("demo.suppress_labels", [])
        if value is None:
            return []
        return [str(label) for label in value]

    @property
    def num_classes(self) -> int:
        return len(self._signs) + (1 if self.rest_enabled else 0)

    @property
    def label_to_idx(self) -> dict[str, int]:
        return dict(self._label_to_idx)

    @property
    def idx_to_label(self) -> list[str]:
        return list(self._idx_to_label)

    @property
    def sign_hash(self) -> str:
        """Changes whenever the sign list/order changes (used to invalidate caches)."""
        payload = "\n".join(self.labels) + f"\nrest={self.rest_enabled}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def feature_dim(self) -> int:
        """Per-frame feature count produced by the keypoint stage."""
        channels = 3 if bool(self.get("keypoints.use_z", False)) else 2
        pose_points = int(self.get("keypoints.pose_points"))
        dim = pose_points * channels
        dim += int(self.get("keypoints.num_hands", 2)) * 21 * channels
        if bool(self.get("keypoints.presence_flags", True)):
            dim += 3
        return dim

    def label_rule(self) -> str:
        return str(self.get("dataset.label_rule", "strip_nonalpha_lower"))

    def normalise(self, folder_name: str) -> str:
        return normalise_label(folder_name, self.label_rule())

    def display(self, label: str) -> str:
        return self.displays.get(label, label)

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {dotted}")
                return default
            node = node[key]
        return node

    def path(self, dotted: str, default: Any = _MISSING) -> Path:
        value = self.get(dotted, default)
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p)

    # Compat aliases so scripts read naturally.
    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    # --------------------------------------------------------------- validation
    def _validate(self) -> None:
        for key in ("project.name", "dataset.signs", "dataset.split_files"):
            self.get(key)  # raises with a useful message if absent

        if not isinstance(self.get("dataset.signs"), list) or not self.get("dataset.signs"):
            raise ConfigError("dataset.signs must be a non-empty list.")

        if self.get("dataset.label_rule", "strip_nonalpha_lower") not in {
            "strip_nonalpha_lower"
        }:
            raise ConfigError(
                "Unsupported dataset.label_rule. Only 'strip_nonalpha_lower' is implemented."
            )

        seen: set[str] = set()
        for i, entry in enumerate(self.get("dataset.signs")):
            if not isinstance(entry, dict) or "label" not in entry or "display" not in entry:
                raise ConfigError(
                    f"dataset.signs[{i}] must be a mapping with 'label' and 'display'."
                )
            label = str(entry["label"])
            if label != self.normalise(label):
                raise ConfigError(
                    f"Sign label {label!r} is not normalised. "
                    f"Expected {self.normalise(label)!r} (rule={self.label_rule()})."
                )
            if label in seen:
                raise ConfigError(f"Duplicate sign label: {label!r}")
            seen.add(label)
            if not str(entry["display"]).strip():
                raise ConfigError(f"Sign {label!r} has an empty display name.")

        if self.rest_enabled and self.rest_label in seen:
            raise ConfigError(
                "A dataset sign is literally called 'rest'; it clashes with the rest class. "
                "Rename it in dataset.signs or disable dataset.rest."
            )

        suppressed = self.get("demo.suppress_labels", [])
        if suppressed is not None and not isinstance(suppressed, list):
            raise ConfigError("demo.suppress_labels must be a list of labels.")
        valid_suppress = set(seen)
        if self.rest_enabled:
            valid_suppress.add(self.rest_label)
        for label in suppressed or []:
            if not isinstance(label, str):
                raise ConfigError(
                    "demo.suppress_labels entries must be strings, "
                    f"got {type(label).__name__}."
                )
            if label not in valid_suppress:
                raise ConfigError(
                    f"demo.suppress_labels contains {label!r}, which is not a dataset "
                    "sign or the rest class. Check the spelling against dataset.signs."
                )

        pose_points = int(self.get("keypoints.pose_points", 25))
        if not 1 <= pose_points <= 33:
            raise ConfigError("keypoints.pose_points must be between 1 and 33.")

        resample = int(self.get("keypoints.resample_frames", 32))
        if resample < 2:
            raise ConfigError("keypoints.resample_frames must be >= 2.")

        input_dim = self.get("model.input_dim", "auto")
        if input_dim not in ("auto", None) and int(input_dim) != self.feature_dim:
            raise ConfigError(
                f"model.input_dim={input_dim} does not match computed feature_dim="
                f"{self.feature_dim}. Use 'auto' or fix the keypoint settings."
            )

        rest_examples = int(self.get("dataset.rest.synthetic_examples", 0))
        if self.rest_enabled and rest_examples < 0:
            raise ConfigError("dataset.rest.synthetic_examples must be >= 0.")

        for split in ("train", "val", "test"):
            if not self.get(f"dataset.split_files.{split}", None):
                raise ConfigError(f"dataset.split_files.{split} is required.")

    def _build_labels(self) -> None:
        self._signs = [
            Sign(label=str(e["label"]), display=str(e["display"]))
            for e in self.get("dataset.signs")
        ]
        labels = [s.label for s in self._signs]
        if self.rest_enabled:
            labels.append(self.rest_label)
        self._idx_to_label = labels
        self._label_to_idx = {label: i for i, label in enumerate(labels)}

    def summary(self) -> str:
        splits = self.get("dataset.split_files")
        return (
            f"config      : {self.config_path}\n"
            f"project     : {self.project_name}\n"
            f"root        : {self.root}\n"
            f"signs       : {len(self._signs)} (+ rest={self.rest_enabled})\n"
            f"classes     : {self.num_classes}\n"
            f"feature_dim : {self.feature_dim}\n"
            f"splits      : {splits}\n"
            f"sign_hash   : {self.sign_hash}"
        )


def load_config(path: str | Path, root: str | Path | None = None) -> Config:
    return Config(path, root=root)


def default_config_path() -> Path:
    """``<repo>/configs/include50.yaml`` relative to this file."""
    return Path(__file__).resolve().parents[2] / "configs" / "include50.yaml"
