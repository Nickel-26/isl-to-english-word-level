"""Fetch individual members of a remote ZIP using HTTP range requests.

Zenodo delivers only ~0.4 MB/s per connection from this network, so pulling
all 42 GB of archives just to keep 958 videos is wasteful. Each ZIP member is
compressed independently, so we can download only the bytes belonging to the
clips we need, then inflate and verify them locally.

This reuses the central-directory metadata recorded by archive_index.py
(header offset, method, CRC, sizes). Nothing is trusted: every file is checked
against its stored CRC-32 and uncompressed size.
"""

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path
from typing import Callable

from .http import HttpError, open_url

LOCAL_SIG = b"PK\x03\x04"

Progress = Callable[[int], None]


def data_offset(url: str, header_offset: int, *, retries: int = 5) -> int:
    """Read a member's local header and return the offset of its data."""
    from .http import read_bytes

    head = read_bytes(
        url, start=header_offset, end=header_offset + 29, retries=retries
    )
    if len(head) < 30 or head[:4] != LOCAL_SIG:
        raise HttpError(
            f"Bad local ZIP header at offset {header_offset} in {url}"
        )
    name_len, extra_len = struct.unpack_from("<HH", head, 26)
    return header_offset + 30 + name_len + extra_len


def fetch_member(
    *,
    url: str,
    header_offset: int,
    method: int,
    crc: int,
    comp_size: int,
    file_size: int,
    dest: Path,
    progress: Progress | None = None,
    chunk_size: int = 1024 * 1024,
    retries: int = 4,
    timeout: float = 180.0,
) -> Path:
    """Download, decompress and verify one ZIP member into ``dest``.

    ``progress`` receives the number of *downloaded* bytes, and a negative
    value if an attempt fails and its byte count has to be rolled back.
    """
    if method not in (0, 8):
        raise HttpError(
            f"{dest.name}: unsupported ZIP compression method {method}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    offset = data_offset(url, header_offset)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        got = 0
        buf = bytearray()
        try:
            headers = {"Range": f"bytes={offset}-{offset + comp_size - 1}"}
            with open_url(url, headers=headers, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if status != 206:
                    raise HttpError(
                        f"{dest.name}: server did not honour the range request "
                        f"(HTTP {status}); refusing to download the whole archive"
                    )
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    got += len(chunk)
                    if progress is not None:
                        progress(len(chunk))

            if len(buf) != comp_size:
                raise HttpError(
                    f"{dest.name}: got {len(buf)} compressed bytes, "
                    f"expected {comp_size}"
                )

            if method == 0:
                data = bytes(buf)
            else:
                data = zlib.decompress(bytes(buf), -15)

            if len(data) != file_size:
                raise HttpError(
                    f"{dest.name}: inflated {len(data)} bytes, expected {file_size}"
                )
            actual_crc = zlib.crc32(data) & 0xFFFFFFFF
            if actual_crc != crc:
                raise HttpError(
                    f"{dest.name}: CRC mismatch (expected {crc:08x}, "
                    f"got {actual_crc:08x})"
                )

            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # retry network / inflate / CRC problems
            last_exc = exc
            if progress is not None and got:
                progress(-got)
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2.0 ** attempt, 20.0))

    raise HttpError(
        f"Failed to fetch {dest.name} after {retries} attempts: {last_exc}"
    )
