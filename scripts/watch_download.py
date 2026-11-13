#!/usr/bin/env python3
"""Write a live, human-readable download status file.

The downloader uses tqdm, whose output contains lines like::

    fetching clips:  23%|██▎       | 2.68G/11.6G [12:34<41:23, 3.72MB/s]

This monitor tails that log, parses the latest progress line, also counts how
many split clips are already on disk, and rewrites a small text file every few
seconds. Watch it with::

    watch -n 2 cat download_progress.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl.config import ConfigError, load_config  # noqa: E402
from isl.splits import all_listed_paths, index_videos_on_disk, load_splits, video_key  # noqa: E402

PROGRESS_RE = re.compile(
    r"(?P<desc>[^:\r\n]+):\s+(?P<pct>\d+)%\|[^|]*\|\s*"
    r"(?P<cur>[0-9.]+[kKMGT]?B?)/(?P<tot>[0-9.]+[kKMGT]?B?)\s*"
    r"\[(?P<elapsed>[^<\]]*)<(?P<eta>[^,\]]*)(?:,\s*(?P<rate>[^\]]*))?\]"
)
SIZE_RE = re.compile(r"([0-9.]+)\s*([KMGT]?)B?", re.IGNORECASE)
UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(text: str) -> float | None:
    m = SIZE_RE.match(text.strip())
    if not m:
        return None
    return float(m.group(1)) * UNITS[m.group(2).upper()]


def human_bytes(n: float) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def tail_text(path: Path, limit: int = 16384) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - limit))
        return fh.read().decode("utf-8", errors="replace")


def latest_progress(tail: str) -> dict | None:
    matches = list(PROGRESS_RE.finditer(tail))
    if not matches:
        return None
    m = matches[-1]
    current = parse_size(m.group("cur"))
    total = parse_size(m.group("tot"))
    return {
        "desc": m.group("desc").strip(),
        "pct": int(m.group("pct")),
        "current": current,
        "total": total,
        "elapsed": m.group("elapsed").strip(),
        "eta": m.group("eta").strip(),
        "rate": (m.group("rate") or "").strip(),
    }


def on_disk_counts(config: Path) -> tuple[int, int]:
    cfg = load_config(config)
    listed = all_listed_paths(load_splits(cfg, strict=True))
    disk = index_videos_on_disk(cfg.path("dataset.raw_dir"))
    present = sum(1 for p in listed if video_key(p) in disk)
    return present, len(listed)


def terminal_state(tail: str) -> str | None:
    if "Done. Next:" in tail:
        return "complete"
    if "clip(s) failed:" in tail or "ERROR:" in tail:
        return "finished with errors"
    return None


def render(
    *,
    progress: dict | None,
    present: int,
    total_clips: int,
    state: str,
    log_path: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "INCLUDE-50 download progress",
        f"updated      : {now}",
        f"status       : {state}",
    ]

    if progress:
        pct = progress["pct"]
        lines.append(f"phase        : {progress['desc']}")
        lines.append(f"progress     : {pct}%")
        if progress["current"] is not None and progress["total"] is not None:
            lines.append(
                f"downloaded   : {human_bytes(progress['current'])} / "
                f"{human_bytes(progress['total'])}"
            )
        if progress["rate"]:
            lines.append(f"speed        : {progress['rate']}")
        lines.append(f"elapsed      : {progress['elapsed']}")
        lines.append(f"eta          : {progress['eta']}")
    else:
        lines.append("phase        : waiting for download output...")

    pct_clips = (100.0 * present / total_clips) if total_clips else 0.0
    lines.append(
        f"clips on disk: {present} / {total_clips} ({pct_clips:.1f}%)"
    )
    lines.append(f"log          : {log_path}")
    lines.append("")
    lines.append("Refresh with: watch -n 2 cat download_progress.txt")
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/tmp/opencode/download.log"),
        help="Downloader log to follow",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "download_progress.txt",
        help="Status text file to rewrite",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "include50.yaml",
        help="Project config (for the clip count)",
    )
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true", help="Write once and exit")
    args = parser.parse_args()

    log_path = args.log.expanduser()

    try:
        present, total_clips = on_disk_counts(args.config)
    except (ConfigError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    idle_since = time.time()
    last_mtime = -1.0
    state = "waiting for downloader"

    while True:
        try:
            mtime = log_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime != last_mtime:
            idle_since = time.time()
            last_mtime = mtime

        tail = tail_text(log_path)
        progress = latest_progress(tail)
        terminal = terminal_state(tail)

        try:
            present, total_clips = on_disk_counts(args.config)
        except (ConfigError, OSError):
            pass

        if terminal:
            state = terminal
        elif progress is None:
            state = "waiting for downloader"
        else:
            state = "running"

        write_atomic(
            args.out,
            render(
                progress=progress,
                present=present,
                total_clips=total_clips,
                state=state,
                log_path=log_path,
            ),
        )

        if args.once:
            return 0

        # Stop once the downloader has clearly finished.
        if terminal and time.time() - idle_since > 10:
            return 0

        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
