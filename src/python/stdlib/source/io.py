"""The ``io`` module: the classes live in ``_io`` so ``open()`` and ``io`` share them."""

import _io
from _io import (
    BlockingIOError, BufferedIOBase, BufferedRandom, BufferedReader, BufferedRWPair,
    BufferedWriter, BytesIO, DEFAULT_BUFFER_SIZE, FileIO, IOBase, IncrementalNewlineDecoder,
    RawIOBase, Reader, SEEK_CUR, SEEK_END, SEEK_SET, StringIO, TextIOBase, TextIOWrapper,
    UnsupportedOperation, Writer, open, open_code, text_encoding,
)

__all__ = [
    "BlockingIOError", "open", "open_code", "IOBase", "RawIOBase", "FileIO", "BytesIO", "StringIO",
    "BufferedIOBase", "BufferedReader", "BufferedWriter", "BufferedRWPair", "BufferedRandom",
    "TextIOBase", "TextIOWrapper", "UnsupportedOperation", "SEEK_SET", "SEEK_CUR", "SEEK_END",
    "DEFAULT_BUFFER_SIZE", "text_encoding", "IncrementalNewlineDecoder", "Reader", "Writer",
]
