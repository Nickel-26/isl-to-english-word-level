#!/usr/bin/env python3
"""Fetch the INCLUDE-50 clips needed by the official split files.

Default mode (``remote``) reads only the needed members of each Zenodo ZIP over
HTTP range requests: it downloads no full archives and needs about 12 GB
instead of 42 GB. Every file is inflate-checked against its ZIP CRC and size.

Fallback mode (``--mode archive``) downloads whole ZIPs, extracts only the
listed clips, then deletes each ZIP. Both modes are resumable: clips already on
disk are skipped, and a failed transfer is retried on the next run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from _bootstrap import add_config_args

from isl.archive_index import (
    archives_for_files,
    human_bytes,
    load_index,
)
from isl.config import ConfigError, load_config
from isl.http import HttpError, download_with_resume
from isl.remote_extract import fetch_member
from isl.splits import (
    all_listed_paths,
    index_videos_on_disk,
    load_splits,
    normalise_rel_path,
    video_key,
)

LOG_NAME = "download_log.jsonl"


def extract_members(
    zip_path: Path,
    wanted: list[str],
    raw_dir: Path,
    *,
    overwrite: bool,
) -> list[str]:
    """Extract only ``wanted`` (relative paths) and verify each file size."""
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        by_name = {normalise_rel_path(i.filename): i for i in zf.infolist() if not i.is_dir()}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise ConfigError(
                f"{zip_path.name}: {len(missing)} expected file(s) missing from ZIP, "
                f"e.g. {missing[:3]}"
            )

        for name in tqdm(wanted, desc=f"  extract {zip_path.name}", unit="file", leave=False):
            info = by_name[name]
            dest = raw_dir / name
            if dest.exists() and not overwrite:
                if dest.stat().st_size == info.file_size:
                    continue
                dest.unlink()

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            try:
                with zf.open(info) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                size = tmp.stat().st_size
                if size != info.file_size:
                    raise ConfigError(
                        f"{name}: extracted {size} bytes, expected {info.file_size}"
                    )
                tmp.replace(dest)
                extracted.append(name)
            finally:
                tmp.unlink(missing_ok=True)
    return extracted


def log_event(cfg, event: dict) -> None:
    path = cfg.path("dataset.manifests_dir") / LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def pick_archives(keys: list[str], only: str | None, max_archives: int | None) -> list[str]:
    selected = keys
    if only:
        needles = [s.strip() for s in only.split(",") if s.strip()]
        selected = [k for k in selected if any(n.lower() in k.lower() for n in needles)]
        if not selected:
            raise ConfigError(f"--only {only!r} matched no needed archive")
    if max_archives is not None:
        selected = selected[:max_archives]
    return selected


def payload_bytes(index: dict, key: str, names: list[str]) -> int:
    """Compressed bytes needed to fetch ``names`` from one archive."""
    entries = index["archives"][key].get("target_entries") or {}
    return sum(int(entries[n]["comp_size"]) for n in names if n in entries)


def remote_tasks(
    index: dict,
    by_archive: dict[str, list[str]],
    raw_dir: Path,
    *,
    overwrite: bool,
) -> tuple[list[tuple[str, str, dict]], list[str]]:
    """Build (archive, member, metadata) tasks for clips not already on disk."""
    todo: list[tuple[str, str, dict]] = []
    missing_meta: list[str] = []
    for key in sorted(by_archive):
        entries = index["archives"][key].get("target_entries") or {}
        for name in by_archive[key]:
            meta = entries.get(name)
            if not meta:
                missing_meta.append(f"{key}:{name}")
                continue
            dest = raw_dir / name
            if (
                not overwrite
                and dest.is_file()
                and dest.stat().st_size == int(meta["file_size"])
            ):
                continue
            todo.append((key, name, meta))
    return todo, missing_meta


def fetch_all(
    index: dict,
    todo: list[tuple[str, str, dict]],
    raw_dir: Path,
    *,
    workers: int,
    desc: str = "fetching clips",
) -> tuple[int, list[str]]:
    """Fetch ZIP members concurrently, reporting byte progress."""
    total = sum(int(meta["comp_size"]) for _, _, meta in todo)
    bar = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        leave=True,
    )
    lock = threading.Lock()

    def progress(n: int) -> None:
        with lock:
            bar.update(n)

    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for key, name, meta in todo:
            info = index["archives"][key]
            fut = pool.submit(
                fetch_member,
                url=str(info["url"]),
                header_offset=int(meta["header_offset"]),
                method=int(meta["method"]),
                crc=int(meta["crc"]),
                comp_size=int(meta["comp_size"]),
                file_size=int(meta["file_size"]),
                dest=raw_dir / name,
                progress=progress,
            )
            futures[fut] = (key, name)
        for fut in as_completed(futures):
            key, name = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as exc:
                errors.append(f"{key}:{name}: {type(exc).__name__}: {exc}")
    bar.close()
    return done, errors


def run_remote_mode(
    cfg,
    index: dict,
    by_archive: dict[str, list[str]],
    needed: list[str],
    raw_dir: Path,
    *,
    overwrite: bool,
    connections: int,
    listed: set[str],
    allow_incomplete: bool,
    started: float,
) -> int:
    todo, missing_meta = remote_tasks(
        index, by_archive, raw_dir, overwrite=overwrite
    )
    if missing_meta:
        raise ConfigError(
            f"{len(missing_meta)} clip(s) have no per-file metadata in the archive "
            "index. Run: python scripts/index_archives.py --refresh"
        )
    if not todo:
        print("\nNo clips left to fetch.")
        return 0

    need_bytes = sum(int(meta["file_size"]) for _, _, meta in todo)
    free = shutil.disk_usage(cfg.root).free
    if free < need_bytes + 1024**3:
        raise ConfigError(
            f"Not enough disk space: {human_bytes(free)} free, need about "
            f"{human_bytes(need_bytes + 1024**3)} for the videos plus margin."
        )

    print(
        f"\nMode: remote — fetching {len(todo)} clip(s) with {connections} "
        f"connections (~{human_bytes(need_bytes)} of video)."
    )
    done, failures = fetch_all(index, todo, raw_dir, workers=connections)

    elapsed = time.time() - started
    remaining = sorted(
        p for p in listed if video_key(p) not in set(index_videos_on_disk(raw_dir))
    )
    print()
    print("=" * 70)
    print(
        f"Fetched {done}/{len(todo)} clip(s) in {elapsed/60:.1f} min | "
        f"clips present: {len(listed) - len(remaining)} / {len(listed)}"
    )
    if remaining:
        print(f"Still missing: {len(remaining)} (run this script again to resume)")
        for p in remaining[:5]:
            print(f"  {p}")

    log_event(
        cfg,
        {
            "event": "remote_fetch",
            "requested": len(todo),
            "fetched": done,
            "failed": len(failures),
            "seconds": round(elapsed, 1),
        },
    )
    if failures:
        print(f"\n{len(failures)} clip(s) failed:", file=sys.stderr)
        for f in failures[:20]:
            print(f"  {f}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        return 1
    if remaining and not allow_incomplete:
        return 1
    print("\nDone. Next: python scripts/verify_dataset.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan only")
    parser.add_argument(
        "--only",
        default=None,
        help="Only these archive key(s), comma-separated or substring (for testing)",
    )
    parser.add_argument(
        "--max-archives", type=int, default=None, help="Stop after N archives"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="Do not delete ZIPs after extraction"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract files that already exist"
    )
    parser.add_argument(
        "--no-md5", action="store_true", help="Skip MD5 verification (faster, less safe)"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even if some split videos are still missing",
    )
    parser.add_argument(
        "--mode",
        choices=("remote", "archive"),
        default=None,
        help=(
            "remote = fetch only the needed ZIP members over HTTP ranges "
            "(default); archive = download whole ZIPs then extract"
        ),
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=None,
        help="Concurrent downloads (default: dataset.download_connections)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        report = load_splits(cfg, strict=True)
        index = load_index(cfg)

        raw_dir = cfg.path("dataset.raw_dir")
        downloads_dir = cfg.path("dataset.downloads_dir")
        raw_dir.mkdir(parents=True, exist_ok=True)
        downloads_dir.mkdir(parents=True, exist_ok=True)

        listed = all_listed_paths(report)
        disk = index_videos_on_disk(raw_dir)
        disk_keys = set(disk)
        missing = sorted(p for p in listed if video_key(p) not in disk_keys)
        present = listed - set(missing)
        print(cfg.summary())
        print()
        print(
            f"Split videos: {len(listed)}  |  already present: {len(present)}  |  "
            f"missing: {len(missing)}"
        )
        variant = sum(1 for k, p in disk.items() if p.suffix.lower() != ".mov")
        if variant:
            print(
                f"Note: {variant} file(s) on disk use a different extension than "
                "the split (e.g. .MP4 instead of .MOV); matched by filename stem."
            )

        if not missing:
            print("\nAll split videos are present. Run scripts/verify_dataset.py next.")
            return 0

        by_archive = archives_for_files(index, missing)
        needed = pick_archives(sorted(by_archive), args.only, args.max_archives)
        if not needed:
            print("No archives selected.", file=sys.stderr)
            return 1

        mode = args.mode or str(cfg.get("dataset.download_mode", "remote"))
        connections = int(
            args.connections or cfg.get("dataset.download_connections", 16)
        )
        total_files = sum(len(by_archive[k]) for k in needed)

        if mode == "remote":
            total_bytes = sum(payload_bytes(index, k, by_archive[k]) for k in needed)
            print(
                f"Archives needed: {len(needed)}  |  clips: {total_files}  |  "
                f"download (compressed clips only): {human_bytes(total_bytes)}"
            )
            for key in needed:
                payload = payload_bytes(index, key, by_archive[key])
                print(
                    f"  {key:<38} {human_bytes(payload):>10}  "
                    f"{len(by_archive[key]):>3} clip(s)"
                )
        else:
            total_bytes = sum(index["archives"][k]["size"] for k in needed)
            print(
                f"Archives to download: {len(needed)}  |  {human_bytes(total_bytes)}  "
                f"|  clips: {total_files}"
            )
            for key in needed:
                info = index["archives"][key]
                print(
                    f"  {key:<38} {human_bytes(info['size']):>10}  "
                    f"{len(by_archive[key]):>3} clip(s)"
                )

        if args.dry_run:
            print(f"\nMode: {mode}. Dry run: nothing downloaded or extracted.")
            return 0

        started = time.time()
        if mode == "remote":
            return run_remote_mode(
                cfg,
                index,
                by_archive,
                needed,
                raw_dir,
                overwrite=args.overwrite,
                connections=connections,
                listed=listed,
                allow_incomplete=args.allow_incomplete,
                started=started,
            )

        verify_md5 = bool(cfg.get("dataset.verify_md5", True)) and not args.no_md5
        keep_archives = args.keep_archives or bool(cfg.get("dataset.keep_archives", False))

        downloaded_bytes = 0
        extracted_total = 0
        failures: list[str] = []

        for i, key in enumerate(needed, 1):
            info = index["archives"][key]
            archive_path = downloads_dir / key
            wanted = [
                p
                for p in by_archive[key]
                if args.overwrite or video_key(p) not in disk_keys
            ]
            if not wanted:
                continue

            print(
                f"\n[{i}/{len(needed)}] {key} — {len(wanted)} clip(s), "
                f"{human_bytes(info['size'])}"
            )

            try:
                free = shutil.disk_usage(cfg.root).free
                if free < info["size"] + 1024**3:
                    raise ConfigError(
                        f"Not enough disk space: {human_bytes(free)} free, "
                        f"need at least {human_bytes(info['size'] + 1024**3)} "
                        f"for {key} plus margin."
                    )

                t0 = time.time()
                download_with_resume(
                    info["url"],
                    archive_path,
                    expected_size=info["size"],
                    expected_md5=info.get("md5") or None,
                    desc=f"  download {key}",
                    verify_md5=verify_md5,
                )
                downloaded_bytes += info["size"]

                extracted = extract_members(
                    archive_path, wanted, raw_dir, overwrite=args.overwrite
                )
                extracted_total += len(extracted)
                disk_keys.update(video_key(n) for n in wanted)

                still_missing = [p for p in wanted if video_key(p) not in disk_keys]
                if still_missing:
                    raise ConfigError(
                        f"{len(still_missing)} clip(s) still missing after extraction, "
                        f"e.g. {still_missing[:3]}. ZIP kept for retry."
                    )

                if not keep_archives:
                    archive_path.unlink(missing_ok=True)
                    archive_path.with_name(archive_path.name + ".part").unlink(
                        missing_ok=True
                    )

                elapsed = time.time() - t0
                print(
                    f"  OK: extracted {len(extracted)} file(s) in {elapsed:.1f}s"
                    + ("" if keep_archives else ", ZIP deleted")
                )
                log_event(
                    cfg,
                    {
                        "event": "archive_ok",
                        "archive": key,
                        "bytes": info["size"],
                        "extracted": len(extracted),
                        "seconds": round(elapsed, 2),
                    },
                )
            except (ConfigError, HttpError, zipfile.BadZipFile, OSError) as exc:
                failures.append(f"{key}: {type(exc).__name__}: {exc}")
                print(f"  FAILED: {exc}", file=sys.stderr)
                log_event(
                    cfg,
                    {
                        "event": "archive_failed",
                        "archive": key,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        print()
        print("=" * 70)
        remaining = sorted(
            p for p in listed if video_key(p) not in set(index_videos_on_disk(raw_dir))
        )
        print(
            f"Downloaded {human_bytes(downloaded_bytes)} in {time.time() - started:.1f}s | "
            f"extracted {extracted_total} file(s)"
        )
        print(f"Clips present: {len(listed) - len(remaining)} / {len(listed)}")
        if remaining:
            print(f"Still missing: {len(remaining)} (run this script again to resume)")
            for p in remaining[:5]:
                print(f"  {p}")
        if failures:
            print(f"\n{len(failures)} archive(s) failed:", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)

        if failures:
            return 1
        if remaining and not args.allow_incomplete:
            return 1
        print("\nDone. Next: python scripts/verify_dataset.py")
        return 0
    except (ConfigError, HttpError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
