#!/usr/bin/env python3
"""Verify that the 958 split videos are present, labelled and readable."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_config_args

from isl.archive_index import human_bytes
from isl.config import ConfigError, load_config
from isl.splits import (
    all_listed_paths,
    index_videos_on_disk,
    label_from_path,
    load_splits,
    video_key,
)


def probe_video(path: Path) -> dict:
    try:
        import cv2
    except ImportError as exc:
        return {"ok": False, "error": f"OpenCV not available: {exc}"}

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"ok": False, "error": "could not open"}
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"ok": False, "error": "could not decode first frame"}
        h, w = frame.shape[:2]
        return {
            "ok": True,
            "width": int(w),
            "height": int(h),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help="Open and decode N random videos with OpenCV (default: 0)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 even if some split videos are missing (useful mid-download)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        report = load_splits(cfg, strict=False)
        raw_dir = cfg.path("dataset.raw_dir")
        listed = all_listed_paths(report)
        disk = index_videos_on_disk(raw_dir)

        print(cfg.summary())
        print()

        errors = list(report.errors)
        warnings = list(report.warnings)

        present: list[str] = []
        empty: list[str] = []
        missing: list[str] = []
        total_bytes = 0

        for rel in sorted(listed):
            p = disk.get(video_key(rel))
            if p is None:
                missing.append(rel)
                continue
            size = p.stat().st_size
            if size <= 0:
                empty.append(rel)
                continue
            present.append(rel)
            total_bytes += size

        # Files on disk that are not in the split (should be none).
        on_disk = {p.relative_to(raw_dir).as_posix() for p in disk.values()}
        listed_keys = {video_key(p) for p in listed}
        extras = sorted(
            p for p in on_disk if video_key(p) not in listed_keys
        )
        extension_variants = sorted(
            p for p in on_disk if p.lower().endswith(".mp4")
        )

        unknown: set[str] = set()
        for p in on_disk:
            try:
                label = label_from_path(p, cfg)
            except ConfigError:
                unknown.add(f"(bad path) {p}")
                continue
            if label not in cfg.label_to_idx:
                unknown.add(label)
        unknown_labels = sorted(unknown)

        if missing:
            errors.append(f"{len(missing)} split video(s) missing from {raw_dir}")
        if empty:
            errors.append(f"{len(empty)} zero-byte video(s)")
        if extras:
            warnings.append(f"{len(extras)} extra video(s) on disk not in the split")
        if unknown_labels:
            errors.append(
                f"{len(unknown_labels)} unknown label(s) on disk: {unknown_labels[:5]}"
            )

        print(f"Split counts     : " + ", ".join(f"{k}={v}" for k, v in report.counts.items()))
        print(f"Clips present    : {len(present)} / {len(listed)}")
        print(f"Missing          : {len(missing)}")
        print(f"Zero-byte        : {len(empty)}")
        print(f"Extras on disk   : {len(extras)}")
        print(f"On-disk size     : {human_bytes(total_bytes)}")
        if extension_variants:
            print(
                f"Extension variants: {len(extension_variants)} .MP4 file(s) matched "
                "to .MOV split entries by filename stem"
            )
        print()

        print(f"{'sign':<14}{'train':>6}{'val':>6}{'test':>6}{'total':>7}")
        print("-" * 39)
        for label in cfg.labels:
            row = report.per_class.get(label, {})
            tr, va, te = row.get("train", 0), row.get("val", 0), row.get("test", 0)
            total = tr + va + te
            flag = "" if total else "  <-- NO CLIPS"
            print(f"{label:<14}{tr:>6}{va:>6}{te:>6}{total:>7}{flag}")
        print("-" * 39)
        totals = [sum(report.per_class.get(l, {}).values()) for l in cfg.labels]
        if totals:
            print(
                f"per-class total: min={min(totals)}, max={max(totals)}, "
                f"mean={sum(totals)/len(totals):.1f}"
            )

        probe_results = []
        if args.probe > 0:
            sample = sorted(present)
            random.Random(int(cfg.get("project.seed", 42))).shuffle(sample)
            sample = sample[: args.probe]
            print(f"\nProbing {len(sample)} video(s) with OpenCV:")
            for rel in sample:
                info = probe_video(disk[video_key(rel)])
                probe_results.append({"path": rel, **info})
                if info.get("ok"):
                    print(
                        f"  OK  {rel}  {info['width']}x{info['height']}  "
                        f"{info['fps']} fps  {info['frames']} frames"
                    )
                else:
                    print(f"  ERR {rel}  {info.get('error')}")
                    errors.append(f"probe failed for {rel}: {info.get('error')}")

        print()
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  - {w}")
        if errors:
            print("Errors:", file=sys.stderr)
            for e in errors[:20]:
                print(f"  - {e}", file=sys.stderr)
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        else:
            print("Dataset verification PASSED.")

        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": str(cfg.config_path),
            "sign_hash": cfg.sign_hash,
            "counts": report.counts,
            "present": len(present),
            "missing_count": len(missing),
            "missing": missing,
            "empty": empty,
            "extras": extras,
            "on_disk_bytes": total_bytes,
            "per_class": report.per_class,
            "probe": probe_results,
            "warnings": warnings,
            "errors": errors,
            "ok": not errors,
        }
        out_path = cfg.path("dataset.manifests_dir") / "dataset_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"Report: {out_path}")

        if errors and not args.allow_missing:
            return 1
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
