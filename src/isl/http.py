"""Small HTTP helpers: retries, byte ranges, resumable downloads."""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Mapping

from tqdm import tqdm
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_HEADERS = {
    "User-Agent": "ivp-isl-course/0.1 (educational project; contact: local user)"
}

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    pass


def _sleep_for_retry(exc: HTTPError, attempt: int, base: float = 1.5) -> None:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = base ** attempt
    else:
        delay = base ** attempt
    delay = min(delay, 60.0) + random.uniform(0, 0.5)
    time.sleep(delay)


def open_url(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    method: str = "GET",
    retries: int = 5,
    timeout: float = 120.0,
):
    """Open a URL, retrying 429/5xx responses with backoff."""
    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        req = Request(url, headers=merged, method=method)
        try:
            return urlopen(req, timeout=timeout)
        except HTTPError as exc:
            last_exc = exc
            if exc.code in RETRY_STATUS and attempt < retries:
                _sleep_for_retry(exc, attempt)
                continue
            raise HttpError(f"HTTP {exc.code} for {url}") from exc
        except URLError as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(1.5 ** attempt, 30) + random.uniform(0, 0.5))
                continue
            raise HttpError(f"Network error for {url}: {exc}") from exc

    raise HttpError(f"Failed after {retries} retries: {url}") from last_exc


def fetch_text(url: str, *, retries: int = 5, timeout: float = 120.0) -> str:
    with open_url(url, retries=retries, timeout=timeout) as resp:
        return resp.read().decode("utf-8-sig")


def read_bytes(
    url: str,
    *,
    start: int | None = None,
    end: int | None = None,
    retries: int = 5,
    timeout: float = 120.0,
) -> bytes:
    """Read a byte range. ``end`` is inclusive, as in HTTP Range."""
    headers = {}
    if start is not None or end is not None:
        if start is None:
            start = 0
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"

    with open_url(url, headers=headers, retries=retries, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        if (start is not None and start > 0) and status == 200:
            raise HttpError(
                f"Server ignored Range request for {url} (got 200 instead of 206)"
            )
        return resp.read()


def md5_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024, desc: str | None = None) -> str:
    h = hashlib.md5()
    total = path.stat().st_size
    with path.open("rb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc or f"md5 {path.name}",
        leave=False,
    ) as bar:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def download_with_resume(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    desc: str | None = None,
    chunk_size: int = 1024 * 1024,
    retries: int = 5,
    timeout: float = 180.0,
    verify_md5: bool = True,
) -> Path:
    """Download ``url`` to ``dest``, resuming from ``dest.part`` if present.

    The partial file is kept on failure so the next run continues where it
    stopped. On success the file is renamed to ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        size_ok = expected_size is None or dest.stat().st_size == expected_size
        if size_ok:
            if verify_md5 and expected_md5:
                actual = md5_file(dest, desc=f"verify {desc or dest.name}")
                if actual.lower() == expected_md5.lower():
                    return dest
                dest.unlink()
            else:
                return dest
        else:
            dest.unlink()

    existing = part.stat().st_size if part.exists() else 0
    if expected_size is not None and existing > expected_size:
        part.unlink()
        existing = 0

    if expected_size is not None and existing == expected_size:
        _finish_download(part, dest, expected_size, expected_md5, verify_md5, desc)
        return dest

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with open_url(url, headers=headers, retries=retries, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if existing and status == 206 else "wb"
        if mode == "wb":
            existing = 0

        total = expected_size
        with part.open(mode) as fh, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or dest.name,
            leave=True,
        ) as bar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))

    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise HttpError(
            f"Incomplete download for {dest.name}: {size} of {expected_size} bytes. "
            f"Partial file kept at {part} for resume."
        )

    _finish_download(part, dest, size, expected_md5, verify_md5, desc)
    return dest


def _finish_download(
    part: Path,
    dest: Path,
    size: int,
    expected_md5: str | None,
    verify_md5: bool,
    desc: str | None,
) -> None:
    if verify_md5 and expected_md5:
        actual = md5_file(part, desc=f"verify {desc or dest.name}")
        if actual.lower() != expected_md5.lower():
            part.unlink(missing_ok=True)
            raise HttpError(
                f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual}. "
                "Deleted the corrupt file; run again to re-download."
            )
    part.replace(dest)


def download_if_missing(
    url: str,
    dest: Path,
    *,
    min_bytes: int = 1,
    expected_size: int | None = None,
    expected_md5: str | None = None,
) -> Path:
    """Download a small file only if it is absent or obviously truncated."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    return download_with_resume(
        url,
        dest,
        expected_size=expected_size,
        expected_md5=expected_md5,
        verify_md5=bool(expected_md5),
    )
