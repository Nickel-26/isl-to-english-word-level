#!/usr/bin/env python3
"""Extract and cache MediaPipe keypoints for every official split clip.

One compressed ``.npz`` per video:
    X            float32 [32, 137]   normalised keypoint sequence
    label        sign label from the config
    split        train / val / test
    source       path relative to data/raw
    sign_hash    invalidates the cache when the sign list changes
    cache_version

The run is resumable: valid cached files are skipped. Progress is written to
``keypoints_progress.txt`` (watch with ``watch -n 2 cat keypoints_progress.txt``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from _bootstrap import ROOT, add_config_args

from isl.config import ConfigError, load_config
from isl.dataset import cache_path
from isl.keypoints import Landmarkers, extract_video
from isl.splits import index_videos_on_disk, load_splits, video_key

_GLOBAL: dict = {}


def _init_worker(config_path: str, root: str | None) -> None:
    cfg = load_config(config_path, root=root)
    _GLOBAL["cfg"] = cfg
    _GLOBAL["landmarkers"] = Landmarkers.from_config(cfg)


def expected_shape(cfg) -> tuple[int, int]:
    return (
        int(cfg.get("keypoints.resample_frames", 32)),
        int(cfg.feature_dim),
    )


def cache_ok(path: Path, cfg) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["cache_version"])
                == int(cfg.get("keypoints.cache_version", 1))
                and str(data["sign_hash"]) == cfg.sign_hash
                and tuple(data["X"].shape) == expected_shape(cfg)
            )
    except (OSError, ValueError, KeyError):
        return False


def cache_problems(path: Path, cfg) -> list[str]:
    """Full validation used by --verify-only: shape, hash, NaN/inf, flags."""
    if not path.is_file():
        return ["missing"]
    problems: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["cache_version"])
            if version != int(cfg.get("keypoints.cache_version", 1)):
                problems.append(f"cache_version {version}")
            if str(data["sign_hash"]) != cfg.sign_hash:
                problems.append("sign_hash mismatch")
            features = data["X"]
            if tuple(features.shape) != expected_shape(cfg):
                problems.append(f"shape {features.shape}")
                return problems
            if not np.isfinite(features).all():
                problems.append("NaN/inf values")
            if bool(cfg.get("keypoints.presence_flags", True)):
                flags = features[:, -3:]
                if not np.isin(flags, (0.0, 1.0)).all():
                    problems.append("presence flags not 0/1")
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"unreadable ({type(exc).__name__}: {exc})")
    return problems


def _process(task: tuple[str, str, str, str, str]) -> dict:
    video_path, dest, label, split, source = task
    cfg = _GLOBAL["cfg"]
    landmarkers: Landmarkers = _GLOBAL["landmarkers"]
    started = time.time()
    try:
        features, meta = extract_video(
            landmarkers,
            Path(video_path),
            pose_points=int(cfg.get("keypoints.pose_points", 25)),
            use_z=bool(cfg.get("keypoints.use_z", False)),
            presence_flags=bool(cfg.get("keypoints.presence_flags", True)),
            resample_frames=int(cfg.get("keypoints.resample_frames", 32)),
            shoulder_visibility=float(cfg.get("keypoints.shoulder_visibility", 0.3)),
        )

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.with_name(dest_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            X=features.astype(np.float32),
            label=str(label),
            split=str(split),
            source=str(source),
            sign_hash=cfg.sign_hash,
            cache_version=np.int64(int(cfg.get("keypoints.cache_version", 1))),
            fps=np.float32(meta["fps"]),
            n_frames=np.int64(meta["n_frames"]),
        )
        tmp.replace(dest_path)
        return {
            "ok": True,
            "label": label,
            "dest": str(dest_path),
            "seconds": time.time() - started,
            "meta": meta,
        }
    except Exception as exc:  # one bad video must not kill the run
        return {
            "ok": False,
            "label": label,
            "dest": dest,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }


def write_status(
    path: Path,
    *,
    done: int,
    total: int,
    ok: int,
    failed: int,
    started: float,
    last: dict | None,
    state: str,
) -> None:
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    remaining = total - done
    eta = remaining / rate if rate > 0 else 0.0
    lines = [
        "INCLUDE-50 keypoint extraction",
        f"updated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"status       : {state}",
        f"progress     : {100.0 * done / total:.1f}% ({done} / {total})",
        f"extracted    : {ok}",
        f"failed       : {failed}",
        f"speed        : {rate:.2f} clips/s",
        f"elapsed      : {int(elapsed)//60:02d}:{int(elapsed)%60:02d}",
        f"eta          : {int(eta)//60:02d}:{int(eta)%60:02d}",
    ]
    if last:
        if last.get("ok"):
            lines.append(
                f"last clip    : {last['label']} ({last['seconds']:.1f}s)"
            )
        else:
            lines.append(f"last error   : {last.get('error', 'unknown')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_entries(cfg, report, raw_dir: Path) -> tuple[list[dict], list[str]]:
    disk = index_videos_on_disk(raw_dir)
    entries: list[dict] = []
    missing: list[str] = []
    for split in ("train", "val", "test"):
        for source in report.splits[split]:
            actual = disk.get(video_key(source))
            if actual is None:
                missing.append(source)
                continue
            label = source.split("/")[1]
            label = cfg.normalise(label)
            entries.append(
                {
                    "source": source,
                    "video_path": str(actual),
                    "label": label,
                    "split": split,
                    "dest": str(cache_path(cfg, label, source)),
                }
            )
    return entries, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: keypoints.workers)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Only this sign label (e.g. hello) or comma-separated labels",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N clips")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract cached clips")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check the cache, do not run MediaPipe",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "keypoints_progress.txt",
        help="Live status text file",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, root=args.root)
        report = load_splits(cfg, strict=True)
        raw_dir = cfg.path("dataset.raw_dir")
        entries, missing = build_entries(cfg, report, raw_dir)

        if missing:
            raise ConfigError(
                f"{len(missing)} split clip(s) are missing on disk, e.g. {missing[:3]}"
            )

        if args.only:
            wanted = {s.strip().lower() for s in args.only.split(",") if s.strip()}
            entries = [e for e in entries if e["label"] in wanted]
            if not entries:
                raise ConfigError(f"--only {args.only!r} matched no clips")
        if args.limit is not None:
            entries = entries[: args.limit]

        cached = [e for e in entries if cache_ok(Path(e["dest"]), cfg)]
        pending = [
            e for e in entries if args.overwrite or not cache_ok(Path(e["dest"]), cfg)
        ]

        print(cfg.summary())
        print()
        print(f"Split clips     : {len(entries)}")
        print(f"Already cached  : {len(cached)}")
        print(f"To extract      : {len(pending)}")
        print(f"Cache dir       : {cfg.path('keypoints.cache_dir')}")
        print(f"Sequence shape  : {expected_shape(cfg)}")
        print(f"Status file     : {args.status_file}")

        if args.verify_only:
            by_split: dict[str, int] = {}
            problems: list[tuple[str, list[str]]] = []
            for entry in entries:
                path = Path(entry["dest"])
                issues = cache_problems(path, cfg)
                if issues:
                    problems.append((str(path), issues))
                else:
                    by_split[entry["split"]] = by_split.get(entry["split"], 0) + 1
            print()
            print(f"Cached by split : {by_split}")
            print(f"Valid cached    : {len(entries) - len(problems)} / {len(entries)}")
            if problems:
                print(f"Problem clips   : {len(problems)}")
                for path, issues in problems[:10]:
                    print(f"  {path}: {', '.join(issues)}")
                if len(problems) > 10:
                    print(f"  ... and {len(problems) - 10} more")
            print(
                "Cache verification "
                + ("PASSED." if not problems else f"has {len(problems)} problem(s).")
            )
            return 0 if not problems else 1

        if not pending:
            print("\nAll selected clips are already cached.")
            write_status(
                args.status_file,
                done=0,
                total=1,
                ok=len(cached),
                failed=0,
                started=time.time(),
                last=None,
                state="complete (nothing to do)",
            )
            return 0

        workers = int(args.workers or cfg.get("keypoints.workers", 2))
        tasks = [
            (e["video_path"], e["dest"], e["label"], e["split"], e["source"])
            for e in pending
        ]

        started = time.time()
        ok = 0
        failures: list[dict] = []
        last: dict | None = None
        write_status(
            args.status_file,
            done=0,
            total=len(tasks),
            ok=0,
            failed=0,
            started=started,
            last=None,
            state="running",
        )

        bar = tqdm(total=len(tasks), unit="clip", desc="extract keypoints")
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(
                processes=max(1, workers),
                initializer=_init_worker,
                initargs=(str(cfg.config_path), str(cfg.root) if cfg.root else None),
            ) as pool:
                for done, result in enumerate(
                    pool.imap_unordered(_process, tasks, chunksize=1), 1
                ):
                    last = result
                    if result["ok"]:
                        ok += 1
                    else:
                        failures.append(result)
                    bar.update(1)
                    if done % 5 == 0 or done == len(tasks):
                        write_status(
                            args.status_file,
                            done=done,
                            total=len(tasks),
                            ok=ok,
                            failed=len(failures),
                            started=started,
                            last=last,
                            state="running",
                        )
        finally:
            bar.close()

        elapsed = time.time() - started
        write_status(
            args.status_file,
            done=len(tasks),
            total=len(tasks),
            ok=ok,
            failed=len(failures),
            started=started,
            last=last,
            state="complete" if not failures else "finished with errors",
        )

        print()
        print("=" * 70)
        print(f"Extracted {ok}/{len(tasks)} clip(s) in {elapsed/60:.1f} min")
        if failures:
            print(f"{len(failures)} clip(s) failed:", file=sys.stderr)
            for failure in failures[:20]:
                print(
                    f"  {failure['dest']}: {failure['error']}", file=sys.stderr
                )
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more", file=sys.stderr)

        report_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sign_hash": cfg.sign_hash,
            "feature_dim": int(cfg.feature_dim),
            "sequence_shape": list(expected_shape(cfg)),
            "selected_clips": len(entries),
            "already_cached": len(cached),
            "requested": len(tasks),
            "extracted": ok,
            "failed": len(failures),
            "seconds": round(elapsed, 1),
            "failures": failures,
        }
        out = cfg.path("dataset.manifests_dir") / "keypoints_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_data, indent=1), encoding="utf-8")
        print(f"Report: {out}")
        print(f"Status: {args.status_file}")

        if failures:
            return 1
        print("\nDone. Next: training (Stage 3).")
        return 0
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
