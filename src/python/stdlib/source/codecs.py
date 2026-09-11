"""Small deterministic codec registry composed from ordinary string and bytes operations."""

_lower = "abcdefghijklmnopqrstuvwxyz"
_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_rot_lower = "nopqrstuvwxyzabcdefghijklm"
_rot_upper = "NOPQRSTUVWXYZABCDEFGHIJKLM"
_rot13 = {}
for source, target in zip(_lower + _upper, _rot_lower + _rot_upper):
    _rot13[source] = target


def encode(value, encoding="utf-8", errors="strict"):
    normalized = encoding.lower().replace("-", "_")
    if normalized in ["rot13", "rot_13"]:
        return "".join(_rot13.get(character, character) for character in value)
    if normalized in ["utf8", "utf_8"]:
        return value.encode("utf-8", errors)
    raise LookupError("unknown encoding: " + encoding)


def decode(value, encoding="utf-8", errors="strict"):
    normalized = encoding.lower().replace("-", "_")
    if normalized in ["rot13", "rot_13"]:
        return "".join(_rot13.get(character, character) for character in value)
    if normalized in ["utf8", "utf_8"]:
        return value.decode("utf-8", errors)
    raise LookupError("unknown encoding: " + encoding)
