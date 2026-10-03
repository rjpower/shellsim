"""JSON encoding and decoding.

The encoder and decoder are written in Python over ordinary string operations so every option
of CPython's ``json`` behaves as documented: hooks, ``default``, ``allow_nan``, ``indent`` and
``separators``. The native ``_json.loads`` is used only as a fast path for hook-free decoding,
and the Python decoder re-parses rejected input to report the error position.
"""

from _json import loads as _native_loads

__all__ = [
    "dump", "dumps", "load", "loads", "JSONDecoder", "JSONDecodeError", "JSONEncoder",
]


class JSONDecodeError(ValueError):
    """Malformed JSON at ``pos`` in ``doc``; ``lineno`` and ``colno`` are one-based."""

    def __init__(self, msg, doc, pos):
        lineno = doc.count("\n", 0, pos) + 1
        colno = pos - doc.rfind("\n", 0, pos)
        ValueError.__init__(self, "%s: line %d column %d (char %d)" % (msg, lineno, colno, pos))
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno


# ---- encoder ----

_ESCAPES = {
    "\\": "\\\\", '"': '\\"', "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t",
}
_INFINITY = float("inf")


def _encode_string(text, ensure_ascii):
    pieces = ['"']
    for char in text:
        escaped = _ESCAPES.get(char)
        if escaped is not None:
            pieces.append(escaped)
            continue
        code = ord(char)
        if code < 0x20:
            pieces.append("\\u%04x" % code)
        elif not ensure_ascii or code < 0x7F:
            pieces.append(char)
        elif code < 0x10000:
            pieces.append("\\u%04x" % code)
        else:
            code -= 0x10000
            pieces.append("\\u%04x\\u%04x" % (0xD800 | (code >> 10), 0xDC00 | (code & 0x3FF)))
    pieces.append('"')
    return "".join(pieces)


def _float_text(value, allow_nan):
    if value != value:
        text = "NaN"
    elif value == _INFINITY:
        text = "Infinity"
    elif value == -_INFINITY:
        text = "-Infinity"
    else:
        return float.__repr__(float(value))
    if not allow_nan:
        raise ValueError("Out of range float values are not JSON compliant: " + repr(value))
    return text


class JSONEncoder:
    """Encode Python values as JSON text; override ``default`` for unsupported types."""

    item_separator = ", "
    key_separator = ": "

    def __init__(self, *, skipkeys=False, ensure_ascii=True, check_circular=True,
                 allow_nan=True, sort_keys=False, indent=None, separators=None, default=None):
        self.skipkeys = skipkeys
        self.ensure_ascii = ensure_ascii
        self.check_circular = check_circular
        self.allow_nan = allow_nan
        self.sort_keys = sort_keys
        if isinstance(indent, int) and not isinstance(indent, bool):
            indent = " " * indent
        self.indent = indent
        if separators is not None:
            self.item_separator, self.key_separator = separators
        elif indent is not None:
            self.item_separator = ","
        if default is not None:
            self.default = default

    def default(self, o):
        raise TypeError("Object of type %s is not JSON serializable" % o.__class__.__name__)

    def encode(self, o):
        if isinstance(o, str):
            return _encode_string(o, self.ensure_ascii)
        return "".join(self.iterencode(o))

    def iterencode(self, o, _one_shot=False):
        markers = {} if self.check_circular else None
        return self._iterencode(o, 0, markers)

    def _iterencode(self, o, level, markers):
        if isinstance(o, str):
            yield _encode_string(o, self.ensure_ascii)
        elif o is None:
            yield "null"
        elif o is True:
            yield "true"
        elif o is False:
            yield "false"
        elif isinstance(o, int):
            yield int.__repr__(int(o))
        elif isinstance(o, float):
            yield _float_text(o, self.allow_nan)
        elif isinstance(o, (list, tuple)):
            yield from self._iterencode_list(o, level, markers)
        elif isinstance(o, dict):
            yield from self._iterencode_dict(o, level, markers)
        else:
            if markers is not None:
                marker = id(o)
                if marker in markers:
                    raise ValueError("Circular reference detected")
                markers[marker] = o
            yield from self._iterencode(self.default(o), level, markers)
            if markers is not None:
                del markers[marker]

    def _newline_indent(self, level):
        if self.indent is None:
            return ""
        return "\n" + self.indent * level

    def _iterencode_list(self, items, level, markers):
        if not items:
            yield "[]"
            return
        if markers is not None:
            marker = id(items)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = items
        inner = self._newline_indent(level + 1)
        separator = self.item_separator + inner
        yield "[" + inner
        first = True
        for value in items:
            if first:
                first = False
            else:
                yield separator
            yield from self._iterencode(value, level + 1, markers)
        yield self._newline_indent(level) + "]"
        if markers is not None:
            del markers[marker]

    def _key_text(self, key):
        if isinstance(key, str):
            return key
        if isinstance(key, float):
            return _float_text(key, self.allow_nan)
        if key is True:
            return "true"
        if key is False:
            return "false"
        if key is None:
            return "null"
        if isinstance(key, int):
            return int.__repr__(int(key))
        if self.skipkeys:
            return None
        raise TypeError("keys must be str, int, float, bool or None, not %s" % key.__class__.__name__)

    def _iterencode_dict(self, mapping, level, markers):
        if not mapping:
            yield "{}"
            return
        if markers is not None:
            marker = id(mapping)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = mapping
        inner = self._newline_indent(level + 1)
        separator = self.item_separator + inner
        yield "{" + inner
        items = mapping.items()
        if self.sort_keys:
            items = sorted(items)
        first = True
        for key, value in items:
            text = self._key_text(key)
            if text is None:
                continue
            if first:
                first = False
            else:
                yield separator
            yield _encode_string(text, self.ensure_ascii)
            yield self.key_separator
            yield from self._iterencode(value, level + 1, markers)
        yield self._newline_indent(level) + "}"
        if markers is not None:
            del markers[marker]


# ---- decoder ----

_WHITESPACE = " \t\n\r"
_CONSTANTS = {"-Infinity": -_INFINITY, "Infinity": _INFINITY, "NaN": float("nan")}
_STRING_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}
_NUMBER_START = "-0123456789"
_DIGITS = "0123456789"


def _skip_whitespace(text, index):
    length = len(text)
    while index < length and text[index] in _WHITESPACE:
        index += 1
    return index


class JSONDecoder:
    """Decode JSON text into Python values, with CPython's hook arguments."""

    def __init__(self, *, object_hook=None, parse_float=None, parse_int=None,
                 parse_constant=None, strict=True, object_pairs_hook=None):
        self.object_hook = object_hook
        self.parse_float = parse_float or float
        self.parse_int = parse_int or int
        self.parse_constant = parse_constant or _CONSTANTS.__getitem__
        self.strict = strict
        self.object_pairs_hook = object_pairs_hook

    def decode(self, s):
        value, end = self.raw_decode(s, _skip_whitespace(s, 0))
        end = _skip_whitespace(s, end)
        if end != len(s):
            raise JSONDecodeError("Extra data", s, end)
        return value

    def raw_decode(self, s, idx=0):
        try:
            return self._scan(s, idx)
        except IndexError:
            raise JSONDecodeError("Expecting value", s, len(s)) from None

    def _scan(self, s, index):
        if index >= len(s):
            raise JSONDecodeError("Expecting value", s, index)
        char = s[index]
        if char == '"':
            return self._parse_string(s, index + 1)
        if char == "{":
            return self._parse_object(s, index + 1)
        if char == "[":
            return self._parse_array(s, index + 1)
        if char == "n" and s[index:index + 4] == "null":
            return None, index + 4
        if char == "t" and s[index:index + 4] == "true":
            return True, index + 4
        if char == "f" and s[index:index + 5] == "false":
            return False, index + 5
        if char in _NUMBER_START:
            return self._parse_number(s, index)
        if char == "N" and s[index:index + 3] == "NaN":
            return self.parse_constant("NaN"), index + 3
        if char == "I" and s[index:index + 8] == "Infinity":
            return self.parse_constant("Infinity"), index + 8
        if char == "-" and s[index:index + 9] == "-Infinity":
            return self.parse_constant("-Infinity"), index + 9
        raise JSONDecodeError("Expecting value", s, index)

    def _parse_number(self, s, index):
        start = index
        length = len(s)
        if s[index] == "-":
            index += 1
        if index >= length or s[index] not in _DIGITS:
            if s[start:start + 9] == "-Infinity":
                return self.parse_constant("-Infinity"), start + 9
            raise JSONDecodeError("Expecting value", s, start)
        if s[index] == "0":
            index += 1
        else:
            while index < length and s[index] in _DIGITS:
                index += 1
        is_float = False
        if index < length and s[index] == "." and index + 1 < length and s[index + 1] in _DIGITS:
            is_float = True
            index += 1
            while index < length and s[index] in _DIGITS:
                index += 1
        if index < length and s[index] in "eE":
            probe = index + 1
            if probe < length and s[probe] in "+-":
                probe += 1
            if probe < length and s[probe] in _DIGITS:
                is_float = True
                index = probe
                while index < length and s[index] in _DIGITS:
                    index += 1
        text = s[start:index]
        if is_float:
            return self.parse_float(text), index
        return self.parse_int(text), index

    def _parse_string(self, s, index):
        pieces = []
        begin = index - 1
        length = len(s)
        while True:
            start = index
            while index < length and s[index] != '"' and s[index] != "\\":
                if self.strict and ord(s[index]) < 0x20:
                    raise JSONDecodeError("Invalid control character at", s, index)
                index += 1
            if index >= length:
                raise JSONDecodeError("Unterminated string starting at", s, begin)
            pieces.append(s[start:index])
            if s[index] == '"':
                return "".join(pieces), index + 1
            index += 1
            if index >= length:
                raise JSONDecodeError("Unterminated string starting at", s, begin)
            escape = s[index]
            if escape == "u":
                code, index = self._parse_unicode_escape(s, index + 1)
                pieces.append(chr(code))
                continue
            replacement = _STRING_ESCAPES.get(escape)
            if replacement is None:
                raise JSONDecodeError("Invalid \\escape", s, index - 1)
            pieces.append(replacement)
            index += 1

    def _parse_unicode_escape(self, s, index):
        digits = s[index:index + 4]
        if len(digits) != 4 or any(char not in "0123456789abcdefABCDEF" for char in digits):
            raise JSONDecodeError("Invalid \\uXXXX escape", s, index - 1)
        code = int(digits, 16)
        index += 4
        if 0xD800 <= code <= 0xDBFF and s[index:index + 2] == "\\u":
            low_digits = s[index + 2:index + 6]
            if len(low_digits) == 4 and all(char in "0123456789abcdefABCDEF" for char in low_digits):
                low = int(low_digits, 16)
                if 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00))
                    index += 6
        if 0xD800 <= code <= 0xDFFF:
            code = 0xFFFD
        return code, index

    def _parse_object(self, s, index):
        pairs = []
        index = _skip_whitespace(s, index)
        if index < len(s) and s[index] == "}":
            return self._finish_object(pairs), index + 1
        while True:
            if index >= len(s) or s[index] != '"':
                raise JSONDecodeError("Expecting property name enclosed in double quotes", s, index)
            key, index = self._parse_string(s, index + 1)
            index = _skip_whitespace(s, index)
            if index >= len(s) or s[index] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, index)
            index = _skip_whitespace(s, index + 1)
            value, index = self._scan(s, index)
            pairs.append((key, value))
            index = _skip_whitespace(s, index)
            if index >= len(s):
                raise JSONDecodeError("Expecting ',' delimiter", s, index)
            if s[index] == "}":
                return self._finish_object(pairs), index + 1
            if s[index] != ",":
                raise JSONDecodeError("Expecting ',' delimiter", s, index)
            index = _skip_whitespace(s, index + 1)
            if index < len(s) and s[index] != '"':
                raise JSONDecodeError("Illegal trailing comma before end of object", s, index - 1) \
                    if s[index] == "}" else \
                    JSONDecodeError("Expecting property name enclosed in double quotes", s, index)

    def _finish_object(self, pairs):
        if self.object_pairs_hook is not None:
            return self.object_pairs_hook(pairs)
        result = dict(pairs)
        if self.object_hook is not None:
            return self.object_hook(result)
        return result

    def _parse_array(self, s, index):
        values = []
        index = _skip_whitespace(s, index)
        if index < len(s) and s[index] == "]":
            return values, index + 1
        while True:
            value, index = self._scan(s, index)
            values.append(value)
            index = _skip_whitespace(s, index)
            if index >= len(s):
                raise JSONDecodeError("Expecting ',' delimiter", s, index)
            if s[index] == "]":
                return values, index + 1
            if s[index] != ",":
                raise JSONDecodeError("Expecting ',' delimiter", s, index)
            index = _skip_whitespace(s, index + 1)
            if index < len(s) and s[index] == "]":
                raise JSONDecodeError("Illegal trailing comma before end of array", s, index - 1)


_default_encoder = JSONEncoder()
_default_decoder = JSONDecoder()


def dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True,
          cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw):
    if (not skipkeys and ensure_ascii and check_circular and allow_nan and cls is None
            and indent is None and separators is None and default is None and not sort_keys
            and not kw):
        return _default_encoder.encode(obj)
    if cls is None:
        cls = JSONEncoder
    return cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
               allow_nan=allow_nan, indent=indent, separators=separators, default=default,
               sort_keys=sort_keys, **kw).encode(obj)


def dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True,
         cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw):
    if cls is None:
        cls = JSONEncoder
    encoder = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                  allow_nan=allow_nan, indent=indent, separators=separators, default=default,
                  sort_keys=sort_keys, **kw)
    for chunk in encoder.iterencode(obj):
        fp.write(chunk)


def detect_encoding(b):
    bstartswith = b.startswith
    if bstartswith((b"\xef\xbb\xbf",)):
        return "utf-8-sig"
    if len(b) >= 4:
        if not b[0]:
            return "utf-16-be" if b[1] else "utf-32-be"
        if not b[1]:
            return "utf-16-le" if b[2] or b[3] else "utf-32-le"
    elif len(b) == 2:
        if not b[0]:
            return "utf-16-be"
        if not b[1]:
            return "utf-16-le"
    return "utf-8"


def loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None,
          parse_constant=None, object_pairs_hook=None, **kw):
    if isinstance(s, str):
        if s.startswith("﻿"):
            raise JSONDecodeError("Unexpected UTF-8 BOM (decode using utf-8-sig)", s, 0)
    else:
        if not isinstance(s, (bytes, bytearray)):
            raise TypeError("the JSON object must be str, bytes or bytearray, not %s"
                            % s.__class__.__name__)
        encoding = detect_encoding(bytes(s))
        if encoding != "utf-8":
            raise JSONDecodeError("only UTF-8 encoded JSON bytes are supported", "", 0)
        s = bytes(s).decode("utf-8")
    if (cls is None and object_hook is None and parse_float is None and parse_int is None
            and parse_constant is None and object_pairs_hook is None and not kw):
        try:
            return _native_loads(s)
        except ValueError:
            # Re-parse in Python so the error names the offending position.
            return _default_decoder.decode(s)
    if cls is None:
        cls = JSONDecoder
    return cls(object_hook=object_hook, parse_float=parse_float, parse_int=parse_int,
               parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw).decode(s)


def load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None,
         parse_constant=None, object_pairs_hook=None, **kw):
    return loads(fp.read(), cls=cls, object_hook=object_hook, parse_float=parse_float,
                 parse_int=parse_int, parse_constant=parse_constant,
                 object_pairs_hook=object_pairs_hook, **kw)
