"""Minimal reader for 8-bit greyscale, non-interlaced PNG images (standard library + numpy).

Used for the EWAP obstacle maps (map.png), which are image-space masks at the video frame size.
Any other PNG variant raises PngError rather than being decoded approximately.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(ValueError):
    """Raised for files that are not valid 8-bit greyscale, non-interlaced PNG."""


def _chunks(data: bytes):
    pos = len(_SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            raise PngError("truncated chunk header")
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:pos + 12 + length]
        if len(body) != length or len(crc) != 4:
            raise PngError(f"truncated {kind!r} chunk")
        if zlib.crc32(kind + body) != struct.unpack(">I", crc)[0]:
            raise PngError(f"CRC mismatch in {kind!r} chunk")
        yield kind, body
        pos += 12 + length


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, width: int, height: int) -> np.ndarray:
    stride = width + 1
    if len(raw) != stride * height:
        raise PngError(f"decompressed size {len(raw)} does not match {width}x{height}")
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    out = np.zeros((height, width), dtype=np.uint8)
    prev = np.zeros(width, dtype=np.int64)
    for y in range(height):
        kind, line = rows[y, 0], rows[y, 1:].astype(np.int64)
        if kind == 0:
            cur = line
        elif kind == 1:                                   # Sub
            cur = np.cumsum(line) & 0xFF
        elif kind == 2:                                   # Up
            cur = (line + prev) & 0xFF
        elif kind in (3, 4):                              # Average, Paeth: sequential
            cur = np.zeros(width, dtype=np.int64)
            for x in range(width):
                left = int(cur[x - 1]) if x else 0
                upper_left = int(prev[x - 1]) if x else 0
                pred = (left + int(prev[x])) // 2 if kind == 3 else _paeth(left, int(prev[x]), upper_left)
                cur[x] = (int(line[x]) + pred) & 0xFF
        else:
            raise PngError(f"unknown filter type {kind} in row {y}")
        out[y] = cur
        prev = cur
    return out


def read_gray8(path: Path) -> np.ndarray:
    """Return the image as a (height, width) uint8 array."""
    data = Path(path).read_bytes()
    if not data.startswith(_SIGNATURE):
        raise PngError(f"{path}: not a PNG file")
    header, idat = None, []
    for kind, body in _chunks(data):
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise PngError(f"{path}: missing IHDR")
    width, height, depth, colour, compression, filtering, interlace = header
    if (depth, colour, compression, filtering, interlace) != (8, 0, 0, 0, 0):
        raise PngError(f"{path}: only 8-bit greyscale non-interlaced PNG is supported "
                       f"(bit depth {depth}, colour type {colour}, interlace {interlace})")
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise PngError(f"{path}: corrupt image data ({exc})") from exc
    return _unfilter(raw, width, height)
