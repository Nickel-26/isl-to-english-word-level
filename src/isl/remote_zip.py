"""List the contents of a remote ZIP using HTTP Range requests.

Zenodo archives are 0.8-1.8 GB each. We only need the ZIP central directory
(the table of contents) to decide which archives contain our 50 signs, so we
read the last ~70 KB plus the central-directory block instead of downloading
45+ GB just to inspect names.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .http import read_bytes

EOCD_SIG = b"PK\x05\x06"
CENTRAL_SIG = b"PK\x01\x02"
MAX_COMMENT = 65535


class ZipFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc: int
    comp_size: int
    file_size: int
    header_offset: int

    @property
    def is_video(self) -> bool:
        return self.name.lower().endswith((".mov", ".mp4", ".avi", ".mkv"))


def list_remote_zip(url: str, size: int, *, retries: int = 6) -> list[ZipEntry]:
    """Return all entries in a remote ZIP, reading only its central directory."""
    if size <= 22:
        raise ZipFormatError(f"File is too small to be a ZIP: {url}")

    tail_len = min(size, MAX_COMMENT + 22 + 4096)
    tail_start = size - tail_len
    tail = read_bytes(url, start=tail_start, end=size - 1, retries=retries)

    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos < 0:
        raise ZipFormatError(f"End-of-central-directory record not found in {url}")

    try:
        (
            _sig,
            _disk,
            _cd_disk,
            _entries_disk,
            entries_total,
            cd_size,
            cd_offset,
            _comment_len,
        ) = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
    except struct.error as exc:
        raise ZipFormatError(f"Malformed EOCD in {url}: {exc}") from exc

    if entries_total == 0xFFFF or cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        raise ZipFormatError(
            f"{url} uses ZIP64, which this lightweight reader does not support."
        )
    if cd_size <= 0 or cd_offset + cd_size > size:
        raise ZipFormatError(
            f"Invalid central directory in {url}: offset={cd_offset}, size={cd_size}"
        )

    cd = read_bytes(url, start=cd_offset, end=cd_offset + cd_size - 1, retries=retries)

    entries: list[ZipEntry] = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != CENTRAL_SIG:
            break
        (
            _sig,
            _ver_made,
            _ver_need,
            flags,
            method,
            _mtime,
            _mdate,
            crc,
            comp_size,
            file_size,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _int_attr,
            _ext_attr,
            header_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", cd, pos)

        if comp_size == 0xFFFFFFFF or file_size == 0xFFFFFFFF or header_offset == 0xFFFFFFFF:
            raise ZipFormatError(f"{url} uses ZIP64 entries, which are not supported.")

        name_start = pos + 46
        name_end = name_start + name_len
        raw_name = cd[name_start:name_end]
        if flags & 0x800:
            name = raw_name.decode("utf-8", errors="replace")
        else:
            try:
                name = raw_name.decode("cp437")
            except UnicodeDecodeError:
                name = raw_name.decode("utf-8", errors="replace")

        entries.append(
            ZipEntry(
                name=name.replace("\\", "/"),
                method=method,
                crc=crc,
                comp_size=comp_size,
                file_size=file_size,
                header_offset=header_offset,
            )
        )
        pos = name_end + extra_len + comment_len

    if len(entries) != entries_total:
        raise ZipFormatError(
            f"{url}: expected {entries_total} entries, parsed {len(entries)}"
        )
    return entries
