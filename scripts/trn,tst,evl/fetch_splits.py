#!/usr/bin/env python3
"""Download the official INCLUDE-50 split files and validate them.

The Google Drive link in the Zenodo record is dead, so the split files are
fetched from AI4Bharat's official inference repository (same 689/77/192 split).
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_config_args

from isl.config import ConfigError, load_config
from isl.http import HttpError, fetch_text
from isl.splits import load_splits, read_split_file, split_file_path


def fetch_one(cfg, split: str, force: bool, allow_mismatch: bool) -> int:
    dest = split_file_path(cfg, split)
    filename = cfg.get(f"dataset.split_files.{split}")
    base = str(cfg.get("dataset.split_source.base_url")).rstrip("/")
    url = f"{base}/{filename}"
    expected = int(cfg.get(f"dataset.split_source.expected_counts.{split}", -1))

    if dest.is_file() and not force:
        count = len(read_split_file(dest))
        if expected < 0 or count == expected:
            print(f"  {split:<5} {count:4d} clips  (cached)  {dest}")
            return count
        print(
            f"  {split:<5} cached count {count} != expected {expected}; re-downloading"
        )

    text = fetch_text(url)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ConfigError(f"Downloaded split is empty: {url}")

    if expected >= 0 and len(lines) != expected:
        msg = f"{split}: downloaded {len(lines)} clips, expected {expected}"
        if not allow_mismatch:
            raise ConfigError(msg + " (use --allow-count-mismatch to override)")
        print(f"  WARNING: {msg}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(dest)
    print(f"  {split:<5} {len(lines):4d} clips  (downloaded)  {dest}")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Keep going if the downloaded count differs from the config",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        print("Fetching INCLUDE-50 split files")
        print(f"  source: {cfg.get('dataset.split_source.base_url')}")
        counts = {
            s: fetch_one(cfg, s, args.force, args.allow_count_mismatch)
            for s in ("train", "val", "test")
        }

        report = load_splits(cfg, strict=True)
        print()
        print(
            f"Validated: {sum(counts.values())} clips, "
            f"{len(cfg.labels)} signs, {len(cfg.labels) + (1 if cfg.rest_enabled else 0)} classes"
        )
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
        print(f"Sign hash: {cfg.sign_hash}")
        return 0
    except (ConfigError, HttpError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
