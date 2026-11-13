#!/usr/bin/env python3
"""Index the Zenodo archives (table of contents only) and map signs to archives.

No videos are downloaded here: for each ZIP we read only its central directory
over HTTP range requests. The result is data/manifests/archive_index.json.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_config_args

from isl.archive_index import build_index, human_bytes, load_index, manifest_path
from isl.config import ConfigError, load_config
from isl.http import HttpError
from isl.splits import load_splits


def print_table(index: dict) -> None:
    archives = index["archives"]
    needed = {k: v for k, v in archives.items() if v.get("target_files")}
    skipped = {k: v for k, v in archives.items() if not v.get("target_files")}

    print()
    print(f"Archives needed ({len(needed)}):")
    print(f"  {'archive':<38} {'size':>10}  {'target':>6}  folders")
    for key in sorted(needed):
        info = needed[key]
        print(
            f"  {key:<38} {human_bytes(info['size']):>10}  "
            f"{len(info['target_files']):>6}  {len(info['target_folders'])}"
        )

    total = sum(v["size"] for v in needed.values())
    print(
        f"  {'TOTAL':<38} {human_bytes(total):>10}  "
        f"{sum(len(v['target_files']) for v in needed.values()):>6}"
    )

    if skipped:
        print()
        print(f"Archives skipped ({len(skipped)}): " + ", ".join(sorted(skipped)))
    print()
    print(f"Split videos mapped: {index['target_videos']}")
    print(f"Sign hash: {index['signs_hash']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--refresh", action="store_true", help="Rescan archives (ignore cached index)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent archive scans (default: dataset.index_workers)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print the cached index, do not rescan",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        print(cfg.summary())
        print()

        if args.summary:
            index = load_index(cfg)
        else:
            report = load_splits(cfg, strict=True)
            print(
                "Split counts: "
                + ", ".join(f"{s}={n}" for s, n in report.counts.items())
            )
            index = build_index(
                cfg, report, force=args.refresh, workers=args.workers
            )

        print_table(index)
        print(f"Manifest: {manifest_path(cfg)}")
        return 0
    except (ConfigError, HttpError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
