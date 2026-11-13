#!/usr/bin/env python3
"""Download the MediaPipe Task models needed for keypoint extraction."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import add_config_args

from isl.archive_index import human_bytes
from isl.config import ConfigError, load_config
from isl.http import HttpError, download_with_resume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--force", action="store_true", help="Re-download models")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        models_dir = cfg.path("keypoints.models_dir")
        models_dir.mkdir(parents=True, exist_ok=True)

        for name in ("hand", "pose"):
            spec = cfg.get(f"keypoints.models.{name}")
            dest = models_dir / str(spec["file"])
            if dest.exists() and not args.force and dest.stat().st_size > 1_000_000:
                print(f"{name:<5} cached   {dest}  ({human_bytes(dest.stat().st_size)})")
                continue
            print(f"{name:<5} downloading {spec['url']}")
            download_with_resume(
                str(spec["url"]),
                dest,
                desc=f"  {dest.name}",
                verify_md5=False,
            )
            size = dest.stat().st_size
            if size < 1_000_000:
                raise ConfigError(f"Downloaded model looks too small: {dest} ({size} bytes)")
            print(f"{name:<5} saved    {dest}  ({human_bytes(size)})")

        print("\nMediaPipe models ready.")
        return 0
    except (ConfigError, HttpError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
