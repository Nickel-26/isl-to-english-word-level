"""Build and read the map of which Zenodo archive holds which sign folders."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import Config, ConfigError
from .http import fetch_text
from .remote_zip import list_remote_zip
from .splits import (
    SplitReport,
    all_listed_paths,
    normalise_rel_path,
    sign_folder_of,
    video_key,
)

ZENODO_API = "https://zenodo.org/api/records/{record}"
INDEX_VERSION = 2  # bump when the per-archive metadata layout changes


def manifest_path(cfg: Config) -> Path:
    return cfg.path("dataset.manifests_dir") / "archive_index.json"


def fetch_record(record: str | int) -> dict:
    url = ZENODO_API.format(record=record)
    return json.loads(fetch_text(url))


def archive_files(record_json: dict) -> list[dict]:
    out = []
    for f in record_json.get("files", []):
        key = f.get("key")
        if not key or not key.lower().endswith(".zip"):
            continue
        checksum = str(f.get("checksum", ""))
        md5 = checksum.split(":", 1)[1] if checksum.lower().startswith("md5:") else ""
        out.append(
            {
                "key": key,
                "size": int(f["size"]),
                "md5": md5,
                "url": f.get("links", {}).get("self"),
            }
        )
    return sorted(out, key=lambda x: x["key"])


def _scan_one(
    archive: dict,
    target_keys: set[str],
) -> dict:
    key = archive["key"]
    entries = list_remote_zip(archive["url"], archive["size"])
    found_files: list[str] = []
    found_entries: dict[str, dict] = {}
    found_folders: set[str] = set()
    video_count = 0
    total_uncompressed = 0

    for entry in entries:
        if not entry.is_video:
            continue
        video_count += 1
        total_uncompressed += entry.file_size
        name = normalise_rel_path(entry.name)
        # Match on folder + stem so a split entry ending in .MOV also finds
        # the same recording stored as .MP4 in the ZIP.
        if video_key(name) in target_keys:
            found_files.append(name)
            found_folders.add(sign_folder_of(name))
            found_entries[name] = {
                "header_offset": entry.header_offset,
                "method": entry.method,
                "crc": entry.crc,
                "comp_size": entry.comp_size,
                "file_size": entry.file_size,
            }

    return {
        "key": key,
        "size": archive["size"],
        "md5": archive["md5"],
        "url": archive["url"],
        "entries": len(entries),
        "video_count": video_count,
        "videos_bytes": total_uncompressed,
        "target_files": sorted(set(found_files)),
        "target_entries": found_entries,
        "target_folders": sorted(found_folders),
    }


def build_index(
    cfg: Config,
    report: SplitReport,
    *,
    force: bool = False,
    workers: int | None = None,
    verbose: bool = True,
) -> dict:
    """Scan every Zenodo archive and record which target videos it contains."""
    if not report.ok:
        raise ConfigError(
            "Split files failed validation:\n  - " + "\n  - ".join(report.errors)
        )

    path = manifest_path(cfg)
    existing: dict = {}
    if path.is_file() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    if existing.get("index_version") != INDEX_VERSION:
        if verbose and existing:
            print("Archive index layout changed; rescanning archives.")
        existing = {}

    if existing.get("signs_hash") not in (None, cfg.sign_hash):
        if verbose:
            print("Sign list changed since the cached index was built; rescanning archives.")
        existing = {}

    record = fetch_record(cfg.get("dataset.zenodo_record"))
    archives = archive_files(record)
    if not archives:
        raise ConfigError("No ZIP files found in the Zenodo record.")

    target_files = all_listed_paths(report)
    target_keys = {video_key(p) for p in target_files}
    key_to_listed = {video_key(p): p for p in target_files}
    target_folders = {sign_folder_of(p) for p in target_files}

    cached = {
        k: v
        for k, v in (existing.get("archives") or {}).items()
        if existing.get("signs_hash") == cfg.sign_hash
    }

    pending: list[dict] = []
    for archive in archives:
        old = cached.get(archive["key"])
        if (
            old
            and old.get("size") == archive["size"]
            and old.get("md5") == archive["md5"]
            and old.get("url") == archive["url"]
        ):
            continue
        pending.append(archive)

    if verbose:
        print(
            f"Zenodo record {cfg.get('dataset.zenodo_record')}: "
            f"{len(archives)} archives, {len(target_files)} target videos, "
            f"{len(target_folders)} sign folders."
        )
        print(f"Archives to scan: {len(pending)} (cached: {len(archives) - len(pending)})")

    results: dict[str, dict] = {}
    errors: list[str] = []
    n_workers = int(workers or cfg.get("dataset.index_workers", 2))

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures = {
            pool.submit(_scan_one, a, target_keys): a
            for a in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            archive = futures[fut]
            try:
                results[archive["key"]] = fut.result()
                if verbose:
                    r = results[archive["key"]]
                    print(
                        f"  [{i:2d}/{len(pending)}] {archive['key']:<38} "
                        f"{r['video_count']:4d} videos, "
                        f"{len(r['target_files']):3d} target clips"
                    )
            except Exception as exc:  # keep going; report at the end
                errors.append(f"{archive['key']}: {type(exc).__name__}: {exc}")
                if verbose:
                    print(f"  [{i:2d}/{len(pending)}] {archive['key']:<38} FAILED: {exc}")

    if errors:
        raise ConfigError(
            "Archive indexing failed for:\n  - " + "\n  - ".join(errors)
        )

    merged = dict(cached)
    merged.update(results)

    # Sanity checks: every listed clip must be found exactly once, by key.
    seen: dict[str, list[str]] = {}
    for key, info in merged.items():
        for name in info.get("target_files", []):
            seen.setdefault(video_key(name), []).append(key)

    missing = sorted(set(key_to_listed) - set(seen))
    duplicates = {k: ks for k, ks in seen.items() if len(ks) > 1}
    if missing:
        examples = [key_to_listed[k] for k in missing[:3]]
        raise ConfigError(
            f"{len(missing)} split clip(s) were not found in any archive, "
            f"e.g. {examples}. The split files and dataset may be out of sync."
        )
    if duplicates:
        key, archives_with = next(iter(duplicates.items()))
        raise ConfigError(
            f"{len(duplicates)} split clip(s) appear in more than one archive, "
            f"e.g. {key_to_listed.get(key, key)} in {archives_with}. "
            "Refusing to guess which copy to use."
        )

    data = {
        "record": str(cfg.get("dataset.zenodo_record")),
        "index_version": INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signs_hash": cfg.sign_hash,
        "n_signs": len(cfg.labels),
        "target_videos": len(target_files),
        "archives": merged,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

    needed = [k for k, info in merged.items() if info.get("target_files")]
    total_bytes = sum(merged[k]["size"] for k in needed)
    if verbose:
        print()
        print(f"Archive index saved to: {path}")
        print(
            f"Archives needed: {len(needed)} / {len(merged)}  |  "
            f"download size: {human_bytes(total_bytes)}"
        )
        print(f"Target clips found: {len(seen)} / {len(target_files)}")
    return data


def load_index(cfg: Config) -> dict:
    path = manifest_path(cfg)
    if not path.is_file():
        raise ConfigError(
            f"Archive index not found: {path}\n"
            "Run: python scripts/index_archives.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("index_version") != INDEX_VERSION:
        raise ConfigError(
            "Archive index uses an older layout. "
            "Run: python scripts/index_archives.py --refresh"
        )
    if data.get("signs_hash") != cfg.sign_hash:
        raise ConfigError(
            "Archive index was built for a different sign list. "
            "Run: python scripts/index_archives.py --refresh"
        )
    return data


def archives_for_files(index: dict, files: Iterable[str]) -> dict[str, list[str]]:
    """Map each archive to the actual ZIP member names needed for ``files``.

    ``files`` may be split paths (``.MOV``); matching is by folder + stem so
    the returned member names are the real entries inside the ZIP (``.MP4``).
    """
    wanted = {video_key(f) for f in files}
    out: dict[str, list[str]] = {}
    for key, info in index["archives"].items():
        hits = [
            n for n in info.get("target_files", []) if video_key(n) in wanted
        ]
        if hits:
            out[key] = hits
    return out


def needed_archive_keys(index: dict, missing_files: Iterable[str]) -> list[str]:
    return sorted(archives_for_files(index, missing_files))


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"
