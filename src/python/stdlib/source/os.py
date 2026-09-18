"""Process and path helpers over shellsim's modeled environment and VFS."""

from _os import chdir, environ, getenv, getcwd
import _shellsim_vfs


def fspath(value):
    if isinstance(value, str):
        return value
    if not hasattr(value, "__fspath__"):
        raise TypeError("expected str or path-like object")
    result = value.__fspath__()
    if not isinstance(result, str):
        raise TypeError("__fspath__() must return str")
    return result


def _components(value):
    value = fspath(value)
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
        first = fspath(first)
        result = first
        for part in parts:
            part = fspath(part)
            if part.startswith("/"):
                result = part
            elif result == "" or result.endswith("/"):
                result += part
            else:
                result += "/" + part
        return result

    def basename(self, value):
        value = fspath(value)
        value = value.rstrip("/")
        return value.split("/")[-1]

    def dirname(self, value):
        value = fspath(value)
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
        value = fspath(value)
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
        return _normpath(fspath(value))

    def abspath(self, value):
        value = fspath(value)
        if value.startswith("/"):
            return _normpath(value)
        return _normpath(self.join(getcwd(), value))

    def exists(self, value):
        return _shellsim_vfs.exists(fspath(value))

    def isfile(self, value):
        return _shellsim_vfs.is_file(fspath(value))

    def isdir(self, value):
        return _shellsim_vfs.is_dir(fspath(value))

    def islink(self, value):
        return _shellsim_vfs.is_symlink(fspath(value))


path = _Path()

F_OK = 0
X_OK = 1
W_OK = 2
R_OK = 4


def access(path, mode):
    path = fspath(path)
    if not _shellsim_vfs.exists(path):
        return False
    if mode == F_OK:
        return True
    permissions = _shellsim_vfs.stat(path)[0]
    if mode & R_OK and permissions & 292 == 0:
        return False
    if mode & W_OK and permissions & 146 == 0:
        return False
    if mode & X_OK and permissions & 73 == 0:
        return False
    return True


def makedirs(name, mode=511, exist_ok=False):
    _shellsim_vfs.mkdir(fspath(name), True, exist_ok)


def listdir(path="."):
    return _shellsim_vfs.list_dir(fspath(path))


def walk(top, topdown=True, onerror=None, followlinks=False):
    top = fspath(top)
    try:
        names = listdir(top)
    except OSError as error:
        if onerror is not None:
            onerror(error)
        return
    directories = []
    files = []
    for name in names:
        child = path.join(top, name)
        if path.isdir(child):
            directories.append(name)
        else:
            files.append(name)
    if topdown:
        yield (top, directories, files)
    for name in directories:
        child = path.join(top, name)
        if followlinks or not path.islink(child):
            for item in walk(child, topdown, onerror, followlinks):
                yield item
    if not topdown:
        yield (top, directories, files)


def remove(path):
    _shellsim_vfs.remove_file(fspath(path))


unlink = remove


def rename(source, destination):
    _shellsim_vfs.rename(fspath(source), fspath(destination))


replace = rename
