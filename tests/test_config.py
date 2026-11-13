"""Basic config/split tests. Run: python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from isl.config import Config, ConfigError, normalise_label  # noqa: E402
from isl.splits import load_splits, split_file_path, video_key  # noqa: E402

CONFIG = ROOT / "configs" / "include50.yaml"


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config(CONFIG)

    def test_sign_count_and_classes(self) -> None:
        self.assertEqual(len(self.cfg.labels), 50)
        self.assertTrue(self.cfg.rest_enabled)
        self.assertEqual(self.cfg.num_classes, 51)

    def test_feature_dim(self) -> None:
        # 25 pose + 2*21 hand points, x/y, plus 3 presence flags.
        self.assertEqual(self.cfg.feature_dim, 137)

    def test_rest_is_last(self) -> None:
        self.assertEqual(self.cfg.idx_to_label[-1], "rest")
        self.assertEqual(self.cfg.label_to_idx["rest"], 50)

    def test_label_normalisation(self) -> None:
        self.assertEqual(normalise_label("54. Cell phone"), "cellphone")
        self.assertEqual(normalise_label("46. you (plural)"), "youplural")
        self.assertEqual(normalise_label("16. train ticket"), "trainticket")
        self.assertEqual(normalise_label("42. T-Shirt"), "tshirt")

    def test_duplicate_sign_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        raw["dataset"]["signs"].append(dict(raw["dataset"]["signs"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs" / "duplicate.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config(path)

    def test_suppress_labels_are_config_driven(self) -> None:
        self.assertEqual(self.cfg.suppress_labels, ["it"])
        # Suppression is presentation-only: the sign stays a model class.
        self.assertIn("it", self.cfg.labels)
        self.assertIn("it", self.cfg.displays)

    def test_unknown_suppress_label_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        raw["demo"]["suppress_labels"] = ["notasign"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs" / "bad_suppress.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config(path)

    def test_non_list_suppress_labels_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        raw["demo"]["suppress_labels"] = "it"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs" / "bad_suppress_type.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config(path)

    def test_video_key_ignores_extension_case(self) -> None:
        self.assertEqual(
            video_key("Home/40. Paint/MVI_8817.MOV"),
            video_key("Home/40. Paint/MVI_8817.MP4"),
        )
        self.assertNotEqual(
            video_key("Home/40. Paint/MVI_8817.MOV"),
            video_key("Home/40. Paint/MVI_8818.MOV"),
        )

    def test_splits_if_downloaded(self) -> None:
        paths = [split_file_path(self.cfg, s) for s in ("train", "val", "test")]
        if not all(p.is_file() for p in paths):
            self.skipTest("split files not downloaded yet")
        report = load_splits(self.cfg, strict=True)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.counts, {"train": 689, "val": 77, "test": 192})
        self.assertEqual(len(report.per_class), 50)

    def test_archive_index_if_present(self) -> None:
        index_path = self.cfg.path("dataset.manifests_dir") / "archive_index.json"
        if not index_path.is_file():
            self.skipTest("archive index not built yet")

        from isl.archive_index import load_index
        from isl.splits import all_listed_paths

        index = load_index(self.cfg)
        archived = {
            name
            for info in index["archives"].values()
            for name in info.get("target_files", [])
        }
        listed = all_listed_paths(load_splits(self.cfg, strict=True))
        self.assertEqual(
            {video_key(p) for p in archived},
            {video_key(p) for p in listed},
        )
        for info in index["archives"].values():
            if info.get("target_files"):
                self.assertEqual(
                    set(info["target_files"]),
                    set(info.get("target_entries", {})),
                )


if __name__ == "__main__":
    unittest.main()
