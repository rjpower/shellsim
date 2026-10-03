"""String constants plus the ``Template`` and ``Formatter`` helpers."""

from _string import (
    ascii_letters,
    ascii_lowercase,
    ascii_uppercase,
    digits,
    hexdigits,
    octdigits,
    printable,
    punctuation,
    whitespace,
)
import re as _re

__all__ = [
    "ascii_letters", "ascii_lowercase", "ascii_uppercase", "capwords", "digits", "hexdigits",
    "octdigits", "printable", "punctuation", "whitespace", "Formatter", "Template",
]


def capwords(s, sep=None):
    return (sep or " ").join(map(str.capitalize, s.split(sep)))


class Template:
    """A string class for supporting $-substitutions."""

    delimiter = "$"
    idpattern = r"(?a:[_a-z][_a-z0-9]*)"
    braceidpattern = None
    flags = _re.IGNORECASE

    def __init__(self, template):
        self.template = template

    def _pattern(self):
        delim = _re.escape(self.delimiter)
        ident = self.idpattern
        bid = self.braceidpattern or self.idpattern
        pattern = (
            delim + "(?:(?P<escaped>" + delim + ")|(?P<named>" + ident + ")|{(?P<braced>" + bid
            + ")}|(?P<invalid>))"
        )
        return _re.compile(pattern, self.flags | _re.VERBOSE)

    def _invalid(self, mo):
        i = mo.start("invalid")
        lines = self.template[:i].splitlines(keepends=True)
        if not lines:
            colno = 1
            lineno = 1
        else:
            colno = i - len("".join(lines[:-1]))
            lineno = len(lines)
        raise ValueError(
            "Invalid placeholder in string: line %d, col %d" % (lineno, colno)
        )

    def substitute(self, mapping=None, /, **kws):
        if mapping is None:
            mapping = kws
        elif kws:
            merged = dict(mapping)
            merged.update(kws)
            mapping = merged

        def convert(mo):
            named = mo.group("named") or mo.group("braced")
            if named is not None:
                return str(mapping[named])
            if mo.group("escaped") is not None:
                return self.delimiter
            if mo.group("invalid") is not None:
                self._invalid(mo)
            raise ValueError("Unrecognized named group in pattern", self._pattern())

        return self._pattern().sub(convert, self.template)

    def safe_substitute(self, mapping=None, /, **kws):
        if mapping is None:
            mapping = kws
        elif kws:
            merged = dict(mapping)
            merged.update(kws)
            mapping = merged

        def convert(mo):
            named = mo.group("named") or mo.group("braced")
            if named is not None:
                try:
                    return str(mapping[named])
                except KeyError:
                    return mo.group()
            if mo.group("escaped") is not None:
                return self.delimiter
            if mo.group("invalid") is not None:
                return mo.group()
            raise ValueError("Unrecognized named group in pattern", self._pattern())

        return self._pattern().sub(convert, self.template)

    def is_valid(self):
        for mo in self._pattern().finditer(self.template):
            if mo.group("invalid") is not None:
                return False
            if mo.group("named") is None and mo.group("braced") is None and mo.group("escaped") is None:
                raise ValueError("Unrecognized named group in pattern", self._pattern())
        return True

    def get_identifiers(self):
        ids = []
        for mo in self._pattern().finditer(self.template):
            named = mo.group("named") or mo.group("braced")
            if named is not None and named not in ids:
                ids.append(named)
            elif named is None and mo.group("invalid") is None and mo.group("escaped") is None:
                raise ValueError("Unrecognized named group in pattern", self._pattern())
        return ids


def _parse_format(format_string):
    """Yield ``(literal_text, field_name, format_spec, conversion)`` as ``Formatter.parse``."""
    literal = []
    index = 0
    length = len(format_string)
    while index < length:
        char = format_string[index]
        if char == "{":
            if index + 1 < length and format_string[index + 1] == "{":
                literal.append("{")
                index += 2
                continue
            end = index + 1
            depth = 1
            while end < length and depth:
                if format_string[end] == "{":
                    depth += 1
                elif format_string[end] == "}":
                    depth -= 1
                end += 1
            if depth:
                raise ValueError("Single '{' encountered in format string")
            field = format_string[index + 1 : end - 1]
            conversion = None
            spec = ""
            spec_index = field.find(":")
            bang = field.find("!")
            if bang != -1 and (spec_index == -1 or bang < spec_index):
                name = field[:bang]
                rest = field[bang + 1 :]
                colon = rest.find(":")
                if colon == -1:
                    conversion = rest
                else:
                    conversion = rest[:colon]
                    spec = rest[colon + 1 :]
                if len(conversion) != 1:
                    raise ValueError("expected ':' after conversion specifier")
            elif spec_index != -1:
                name = field[:spec_index]
                spec = field[spec_index + 1 :]
            else:
                name = field
            yield ("".join(literal), name, spec, conversion)
            literal = []
            index = end
        elif char == "}":
            if index + 1 < length and format_string[index + 1] == "}":
                literal.append("}")
                index += 2
                continue
            raise ValueError("Single '}' encountered in format string")
        else:
            literal.append(char)
            index += 1
    if literal:
        yield ("".join(literal), None, None, None)


class Formatter:
    def format(self, format_string, /, *args, **kwargs):
        return self.vformat(format_string, args, kwargs)

    def vformat(self, format_string, args, kwargs):
        used_args = set()
        result, _ = self._vformat(format_string, args, kwargs, used_args, 2)
        self.check_unused_args(used_args, args, kwargs)
        return result

    def _vformat(self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0):
        if recursion_depth < 0:
            raise ValueError("Max string recursion exceeded")
        result = []
        for literal_text, field_name, format_spec, conversion in self.parse(format_string):
            if literal_text:
                result.append(literal_text)
            if field_name is None:
                continue
            if field_name == "":
                if auto_arg_index is False:
                    raise ValueError(
                        "cannot switch from manual field specification to automatic field numbering"
                    )
                field_name = str(auto_arg_index)
                auto_arg_index += 1
            elif field_name.isdigit():
                if auto_arg_index:
                    raise ValueError(
                        "cannot switch from manual field specification to automatic field numbering"
                    )
                auto_arg_index = False
            obj, arg_used = self.get_field(field_name, args, kwargs)
            used_args.add(arg_used)
            obj = self.convert_field(obj, conversion)
            format_spec, auto_arg_index = self._vformat(
                format_spec, args, kwargs, used_args, recursion_depth - 1, auto_arg_index
            )
            result.append(self.format_field(obj, format_spec))
        return "".join(result), auto_arg_index

    def get_value(self, key, args, kwargs):
        if isinstance(key, int):
            return args[key]
        return kwargs[key]

    def check_unused_args(self, used_args, args, kwargs):
        pass

    def format_field(self, value, format_spec):
        return format(value, format_spec)

    def convert_field(self, value, conversion):
        if conversion is None:
            return value
        if conversion == "s":
            return str(value)
        if conversion == "r":
            return repr(value)
        if conversion == "a":
            return ascii(value)
        raise ValueError("Unknown conversion specifier " + conversion)

    def parse(self, format_string):
        return _parse_format(format_string)

    def get_field(self, field_name, args, kwargs):
        end = 0
        length = len(field_name)
        while end < length and field_name[end] not in ".[":
            end += 1
        first = field_name[:end]
        key = int(first) if first.isdigit() else first
        obj = self.get_value(key, args, kwargs)
        rest = field_name[end:]
        while rest:
            if rest[0] == ".":
                stop = 1
                while stop < len(rest) and rest[stop] not in ".[":
                    stop += 1
                obj = getattr(obj, rest[1:stop])
                rest = rest[stop:]
            elif rest[0] == "[":
                stop = rest.find("]")
                if stop == -1:
                    raise ValueError("Missing ']' in format string")
                item = rest[1:stop]
                obj = obj[int(item) if item.isdigit() else item]
                rest = rest[stop + 1 :]
            else:
                raise ValueError("Only '.' or '[' may follow ']' in format field specifier")
        return obj, key
