"""Process and path helpers over shellsim's modeled environment and VFS."""

from _os import environ, getenv, getcwd
import _shellsim_vfs


def _components(value):
    return [part for part in value.split("/") if part != "" and part != "."]


def _normpath(value):
    absolute = value.startswith("/")
    parts = []
    for part in _components(value):
        if part == "..":
            if len(parts) > 0 and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append(part)
        else:
            parts.append(part)
    result = "/".join(parts)
    if absolute:
        return "/" + result
    if result == "":
        return "."
    return result


class _Path:
    def join(self, first, *parts):
        result = first
        for part in parts:
            if part.startswith("/"):
                result = part
            elif result == "" or result.endswith("/"):
                result += part
            else:
                result += "/" + part
        return result

    def basename(self, value):
        value = value.rstrip("/")
        return value.split("/")[-1]

    def dirname(self, value):
        value = value.rstrip("/")
        parts = value.split("/")
        if len(parts) <= 1:
            return ""
        result = "/".join(parts[:-1])
        if result == "" and value.startswith("/"):
            return "/"
        return result

    def split(self, value):
        return (self.dirname(value), self.basename(value))

    def splitext(self, value):
        directory = self.dirname(value)
        name = self.basename(value)
        position = len(name) - 1
        while position > 0 and name[position] != ".":
            position -= 1
        if position <= 0:
            return (value, "")
        root = name[:position]
        if directory != "":
            root = self.join(directory, root)
        return (root, name[position:])

    def normpath(self, value):
        return _normpath(value)

    def abspath(self, value):
        if value.startswith("/"):
            return _normpath(value)
        return _normpath(self.join(getcwd(), value))

    def exists(self, value):
        return _shellsim_vfs.exists(value)

    def isfile(self, value):
        return _shellsim_vfs.is_file(value)

    def isdir(self, value):
        return _shellsim_vfs.is_dir(value)


path = _Path()


def makedirs(name, mode=511, exist_ok=False):
    _shellsim_vfs.mkdir(name, True, exist_ok)
