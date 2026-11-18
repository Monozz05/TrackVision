"""The minimal greyscale PNG reader, against an independent reference encoder."""

import struct
import zlib

import numpy as np
import pytest

from fixtures import synthetic_png
from trackvision.io.png import PngError, read_gray8

IMAGE = np.random.default_rng(0).integers(0, 256, size=(7, 11), dtype=np.uint8)


@pytest.mark.parametrize("filters", [[k] * 7 for k in range(5)] + [[0, 1, 2, 3, 4, 2, 1]],
                         ids=["none", "sub", "up", "average", "paeth", "mixed"])
def test_round_trip_for_every_filter_type(tmp_path, filters):
    path = tmp_path / "x.png"
    path.write_bytes(synthetic_png.encode_gray8(IMAGE, filters))
    np.testing.assert_array_equal(read_gray8(path), IMAGE)


def _with_header(tmp_path, depth, colour, interlace=0):
    header = struct.pack(">IIBBBBB", 2, 2, depth, colour, 0, 0, interlace)
    data = (synthetic_png.SIGNATURE + synthetic_png.chunk(b"IHDR", header)
            + synthetic_png.chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + synthetic_png.chunk(b"IEND", b""))
    path = tmp_path / "x.png"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("depth, colour, interlace", [(16, 0, 0), (8, 2, 0), (8, 6, 0), (8, 0, 1)],
                         ids=["16-bit", "rgb", "rgba", "interlaced"])
def test_unsupported_variants_raise(tmp_path, depth, colour, interlace):
    with pytest.raises(PngError):
        read_gray8(_with_header(tmp_path, depth, colour, interlace))


def test_corrupt_crc_raises(tmp_path):
    data = bytearray(synthetic_png.encode_gray8(IMAGE))
    data[20] ^= 0xFF                                   # inside the IHDR body (bytes 16-28)
    path = tmp_path / "x.png"
    path.write_bytes(bytes(data))
    with pytest.raises(PngError, match="CRC"):
        read_gray8(path)


def test_non_png_raises(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"not a png")
    with pytest.raises(PngError):
        read_gray8(path)
