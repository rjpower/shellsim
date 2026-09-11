"""JSON facade composed around shellsim's bounded native codec."""

from _json import dumps, loads


def dump(value, stream, sort_keys=False, separators=None, indent=None):
    if separators is None:
        text = dumps(value, sort_keys=sort_keys, indent=indent)
    else:
        text = dumps(value, sort_keys=sort_keys, separators=separators, indent=indent)
    stream.write(text)


def load(stream):
    return loads(stream.read())
