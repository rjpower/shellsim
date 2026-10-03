"""Bounded byte compression over shellsim's narrow native codec core.

The native core compresses and decompresses whole zlib-format streams. The streaming objects
buffer their input and run the whole-stream codec when the stream is complete, so results match
a one-shot call while keeping the native surface small.
"""

from _zlib import compress as _compress, crc32, decompress as _decompress

__all__ = [
    "DEFLATED", "DEF_BUF_SIZE", "DEF_MEM_LEVEL", "MAX_WBITS", "ZLIB_RUNTIME_VERSION",
    "ZLIB_VERSION", "Z_BEST_COMPRESSION", "Z_BEST_SPEED", "Z_BLOCK", "Z_DEFAULT_COMPRESSION",
    "Z_DEFAULT_STRATEGY", "Z_FILTERED", "Z_FINISH", "Z_FIXED", "Z_FULL_FLUSH", "Z_HUFFMAN_ONLY",
    "Z_NO_COMPRESSION", "Z_NO_FLUSH", "Z_PARTIAL_FLUSH", "Z_RLE", "Z_SYNC_FLUSH", "Z_TREES",
    "adler32", "compress", "compressobj", "crc32", "decompress", "decompressobj", "error",
]

ZLIB_VERSION = "1.3.1"
ZLIB_RUNTIME_VERSION = ZLIB_VERSION
MAX_WBITS = 15
DEFLATED = 8
DEF_MEM_LEVEL = 8
DEF_BUF_SIZE = 16384
Z_NO_COMPRESSION = 0
Z_BEST_SPEED = 1
Z_BEST_COMPRESSION = 9
Z_DEFAULT_COMPRESSION = -1
Z_DEFAULT_STRATEGY = 0
Z_FILTERED = 1
Z_HUFFMAN_ONLY = 2
Z_RLE = 3
Z_FIXED = 4
Z_NO_FLUSH = 0
Z_PARTIAL_FLUSH = 1
Z_SYNC_FLUSH = 2
Z_FULL_FLUSH = 3
Z_FINISH = 4
Z_BLOCK = 5
Z_TREES = 6


class error(Exception):
    pass


def _check_wbits(wbits):
    if wbits != MAX_WBITS and not 9 <= wbits <= 15:
        raise error("Invalid initialization option: only zlib-format streams are supported")


def compress(data, /, level=Z_DEFAULT_COMPRESSION, wbits=MAX_WBITS):
    _check_wbits(wbits)
    try:
        return _compress(bytes(data), level)
    except ValueError as exc:
        raise error(str(exc)) from None


def decompress(data, /, wbits=MAX_WBITS, bufsize=DEF_BUF_SIZE):
    if bufsize < 0:
        raise ValueError("bufsize must be non-negative")
    _check_wbits(wbits)
    try:
        return _decompress(bytes(data))
    except ValueError as exc:
        raise error("Error -3 while decompressing data: " + str(exc)) from None


def adler32(data, value=1, /):
    low = value & 0xFFFF
    high = (value >> 16) & 0xFFFF
    for byte in bytes(data):
        low = (low + byte) % 65521
        high = (high + low) % 65521
    return (high << 16) | low


class _Compress:
    def __init__(self, level, wbits):
        self._level = level
        self._wbits = wbits
        self._pending = bytearray()
        self._finished = False

    def compress(self, data, /):
        if self._finished:
            raise error("Error -2 while compressing data: inconsistent stream state")
        self._pending += bytes(data)
        return b""

    def flush(self, mode=Z_FINISH, /):
        if mode == Z_NO_FLUSH or self._finished:
            return b""
        if mode != Z_FINISH:
            # Partial flushes have no streaming equivalent; the data goes out at Z_FINISH.
            return b""
        self._finished = True
        return compress(bytes(self._pending), self._level, self._wbits)

    def copy(self):
        if self._finished:
            raise ValueError("Inconsistent stream state")
        clone = _Compress(self._level, self._wbits)
        clone._pending = bytearray(self._pending)
        return clone

    __copy__ = copy

    def __deepcopy__(self, memo):
        return self.copy()


class _Decompress:
    def __init__(self, wbits):
        self._wbits = wbits
        self._pending = bytearray()
        self.unused_data = b""
        self.unconsumed_tail = b""
        self.eof = False

    def decompress(self, data, /, max_length=0):
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        data = bytes(data)
        if self.eof:
            self.unused_data += data
            return b""
        self._pending += data
        try:
            result = _decompress(bytes(self._pending))
        except ValueError:
            # The stream is incomplete so far; a corrupt stream surfaces at flush().
            return b""
        self.eof = True
        self._pending = bytearray()
        return result

    def flush(self, length=DEF_BUF_SIZE, /):
        if length <= 0:
            raise ValueError("length must be greater than zero")
        if self.eof or not self._pending:
            return b""
        result = decompress(bytes(self._pending), self._wbits)
        self.eof = True
        self._pending = bytearray()
        return result

    def copy(self):
        clone = _Decompress(self._wbits)
        clone._pending = bytearray(self._pending)
        clone.unused_data = self.unused_data
        clone.eof = self.eof
        return clone

    __copy__ = copy

    def __deepcopy__(self, memo):
        return self.copy()


def compressobj(level=Z_DEFAULT_COMPRESSION, method=DEFLATED, wbits=MAX_WBITS,
                memLevel=DEF_MEM_LEVEL, strategy=Z_DEFAULT_STRATEGY, zdict=None):
    if method != DEFLATED:
        raise error("Invalid initialization option")
    if not -1 <= level <= 9:
        raise error("Bad compression level")
    if zdict is not None:
        raise error("preset dictionaries are not supported")
    _check_wbits(wbits)
    return _Compress(level, wbits)


def decompressobj(wbits=MAX_WBITS, zdict=None):
    if zdict is not None:
        raise error("preset dictionaries are not supported")
    _check_wbits(wbits)
    return _Decompress(wbits)
