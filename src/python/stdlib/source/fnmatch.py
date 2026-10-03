"""Shell-style wildcard matching: ``*``, ``?`` and ``[seq]`` compiled to regular expressions."""

import os
import posixpath
import re

__all__ = ["filter", "fnmatch", "fnmatchcase", "translate"]

_cache = {}
_CACHE_LIMIT = 256


def _compile_pattern(pattern):
    compiled = _cache.get(pattern)
    if compiled is None:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        compiled = re.compile(translate(pattern)).match
        _cache[pattern] = compiled
    return compiled


def fnmatch(name, pat):
    name = os.path.normcase(name)
    pat = os.path.normcase(pat)
    return fnmatchcase(name, pat)


def fnmatchcase(name, pat):
    return _compile_pattern(pat)(name) is not None


def filter(names, pat):
    match = _compile_pattern(os.path.normcase(pat))
    if os.path is posixpath:
        return [name for name in names if match(name)]
    return [name for name in names if match(os.path.normcase(name))]


def translate(pat):
    """The regular expression for ``pat``; ``*`` and ``?`` match path separators too."""
    index = 0
    length = len(pat)
    parts = []
    while index < length:
        char = pat[index]
        index += 1
        if char == "*":
            if not parts or parts[-1] != "*":
                parts.append("*")
        elif char == "?":
            parts.append(".")
        elif char == "[":
            end = index
            if end < length and pat[end] == "!":
                end += 1
            if end < length and pat[end] == "]":
                end += 1
            while end < length and pat[end] != "]":
                end += 1
            if end >= length:
                parts.append("\\[")
            else:
                content = pat[index:end]
                index = end + 1
                if content[0] == "!":
                    content = "^" + content[1:]
                elif content[0] == "^":
                    content = "\\" + content
                content = content.replace("\\", "\\\\")
                parts.append("[" + content + "]")
        else:
            parts.append(re.escape(char))
    result = []
    for part in parts:
        if part == "*":
            result.append(".*")
        else:
            result.append(part)
    return r"(?s:%s)\Z" % "".join(result)
