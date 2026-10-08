"""Bounded ASCII and base64 helpers used by pure Python packages."""

from _base64 import b64decode as _b64decode
from _base64 import b64encode as _b64encode


class Error(ValueError):
    """Malformed binary-to-ASCII data."""


class Incomplete(Exception):
    """Incomplete binary-to-ASCII data."""


def a2b_base64(data, *, strict_mode=False):
    if isinstance(data, str):
        data = data.encode("ascii")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("a2b_base64 requires a bytes-like object or ASCII string")
    if not strict_mode:
        alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        data = bytes(value for value in data if value in alphabet)
    try:
        return _b64decode(bytes(data))
    except ValueError as error:
        raise Error(str(error)) from error


def b2a_base64(data, *, newline=True):
    result = _b64encode(data)
    return result + b"\n" if newline else result
