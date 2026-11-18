"""Reference encoder for 8-bit greyscale PNG with a chosen filter type per row, written
independently of trackvision.io.png so that the reader is tested against something other than
itself."""

import struct
import zlib

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def _filter_row(kind: int, row: list[int], prev: list[int]) -> list[int]:
    out = []
    for x, value in enumerate(row):
        left = row[x - 1] if x else 0
        up = prev[x]
        upper_left = prev[x - 1] if x else 0
        pred = (0, left, up, (left + up) // 2, _paeth(left, up, upper_left))[kind]
        out.append((value - pred) % 256)
    return out


def encode_gray8(image: np.ndarray, filters=None) -> bytes:
    """PNG bytes for a (height, width) uint8 array; `filters` gives one filter type per row."""
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape
    filters = [0] * height if filters is None else list(filters)
    raw, prev = bytearray(), [0] * width
    for y in range(height):
        row = image[y].tolist()
        raw.append(filters[y])
        raw.extend(_filter_row(filters[y], row, prev))
        prev = row
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))
