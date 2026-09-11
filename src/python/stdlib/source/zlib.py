"""Checksum support with compression explicitly gated on real bytes."""

from _zlib import crc32


def compress(data, level=-1):
    raise RuntimeError("zlib.compress requires byte-preserving PyBytes support")


def decompress(data):
    raise RuntimeError("zlib.decompress requires byte-preserving PyBytes support")
