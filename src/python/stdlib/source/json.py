"""JSON facade composed around shellsim's bounded native codec."""

from _json import dumps, loads

# The bounded native decoder reports ValueError. Using the same class object
# keeps normal ``except json.JSONDecodeError`` code correct without a wrapper.
JSONDecodeError = ValueError


def dump(value, stream, sort_keys=False, separators=None, indent=None, ensure_ascii=True):
    if separators is None:
        text = dumps(value, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii)
    else:
        text = dumps(
            value,
            sort_keys=sort_keys,
            separators=separators,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )
    stream.write(text)


def load(stream):
    return loads(stream.read())
