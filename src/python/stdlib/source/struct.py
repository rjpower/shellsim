"""Fixed-width binary records over shellsim's deterministic native core."""

import _struct

__all__ = ["calcsize", "pack", "pack_into", "unpack", "unpack_from", "iter_unpack", "Struct", "error"]


class error(Exception):
    """A bad format string, a value that does not fit its format, or a short buffer."""


def calcsize(format):
    try:
        return _struct.calcsize(format)
    except (ValueError, OverflowError) as exc:
        raise error(str(exc)) from None


def pack(format, *values):
    try:
        return _struct.pack(format, *values)
    except (ValueError, OverflowError) as exc:
        raise error(str(exc)) from None


def unpack(format, buffer):
    try:
        return _struct.unpack(format, bytes(buffer))
    except (ValueError, OverflowError) as exc:
        raise error(str(exc)) from None


def unpack_from(format, buffer, offset=0):
    size = calcsize(format)
    data = bytes(buffer)[offset:offset + size]
    if len(data) < size:
        raise error("unpack_from requires a buffer of at least %d bytes" % (size + offset))
    return unpack(format, data)


def pack_into(format, buffer, offset, *values):
    packed = pack(format, *values)
    if offset < 0:
        offset += len(buffer)
    if offset < 0 or offset + len(packed) > len(buffer):
        raise error("pack_into requires a buffer of at least %d bytes" % (offset + len(packed)))
    buffer[offset:offset + len(packed)] = packed


def iter_unpack(format, buffer):
    size = calcsize(format)
    if size == 0:
        raise error("cannot iteratively unpack with a struct of length 0")
    data = bytes(buffer)
    if len(data) % size != 0:
        raise error("iterative unpacking requires a buffer of a multiple of %d bytes" % size)
    for start in range(0, len(data), size):
        yield unpack(format, data[start:start + size])


class Struct:
    """A compiled format string with the module functions as methods."""

    def __init__(self, format):
        if isinstance(format, bytes):
            format = format.decode("ascii")
        self.format = format
        self.size = calcsize(format)

    def pack(self, *values):
        return pack(self.format, *values)

    def pack_into(self, buffer, offset, *values):
        return pack_into(self.format, buffer, offset, *values)

    def unpack(self, buffer):
        return unpack(self.format, bytes(buffer))

    def unpack_from(self, buffer, offset=0):
        return unpack_from(self.format, buffer, offset)

    def iter_unpack(self, buffer):
        return iter_unpack(self.format, buffer)

    def __repr__(self):
        return "Struct(%r)" % (self.format,)
