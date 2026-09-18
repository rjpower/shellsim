"""Small Path facade over shellsim's modeled VFS."""

import _shellsim_vfs
import os


def _join(left, right):
    if right.startswith("/"):
        return right
    if left == "" or left.endswith("/"):
        return left + right
    return left + "/" + right


def _parts(value):
    return [part for part in os.path.abspath(value).split("/") if part]


class _StatResult:
    def __init__(self, mode, size):
        self.st_mode = mode
        self.st_size = size


class Path:
    def __init__(self, value="."):
        self._path = str(value)

    def __str__(self):
        return self._path

    def __repr__(self):
        return "PosixPath('" + self._path + "')"

    def __fspath__(self):
        return self._path

    def __truediv__(self, other):
        return Path(_join(self._path, str(other)))

    def __eq__(self, other):
        return isinstance(other, Path) and self._path == other._path

    def joinpath(self, other):
        return self / other

    @property
    def name(self):
        parts = self._path.rstrip("/").split("/")
        return parts[-1]

    @property
    def parent(self):
        parts = self._path.rstrip("/").split("/")
        if len(parts) <= 1:
            return Path(".")
        parent = "/".join(parts[:-1])
        if parent == "" and self._path.startswith("/"):
            parent = "/"
        return Path(parent)

    @property
    def parts(self):
        absolute = self._path.startswith("/")
        parts = tuple(part for part in self._path.split("/") if part and part != ".")
        if absolute:
            return ("/",) + parts
        return parts

    @property
    def suffix(self):
        name = self.name
        index = len(name) - 1
        while index >= 0 and name[index] != ".":
            index -= 1
        if index <= 0:
            return ""
        return name[index:]

    @property
    def stem(self):
        suffix = self.suffix
        if suffix == "":
            return self.name
        return self.name[:-len(suffix)]

    def exists(self):
        return _shellsim_vfs.exists(self._path)

    def is_file(self):
        return _shellsim_vfs.is_file(self._path)

    def is_dir(self):
        return _shellsim_vfs.is_dir(self._path)

    def mkdir(self, parents=False, exist_ok=False):
        _shellsim_vfs.mkdir(self._path, parents, exist_ok)

    def glob(self, pattern):
        return [Path(value) for value in _shellsim_vfs.glob(_join(self._path, pattern))]

    def rglob(self, pattern):
        matches = []
        pending = [self]
        while pending:
            directory = pending.pop()
            matches.extend(directory.glob(pattern))
            for child in directory.glob("*"):
                if child.is_dir():
                    pending.append(child)
        return matches

    def iterdir(self):
        return self.glob("*")

    def resolve(self):
        return Path(os.path.abspath(self._path))

    def relative_to(self, other):
        own = _parts(self._path)
        base = _parts(str(other))
        if own[:len(base)] != base:
            raise ValueError(str(self) + " is not in the subpath of " + str(other))
        remainder = own[len(base):]
        return Path("." if len(remainder) == 0 else "/".join(remainder))

    def stat(self):
        mode, size = _shellsim_vfs.stat(self._path)
        return _StatResult(mode, size)

    @classmethod
    def cwd(cls):
        return cls(os.getcwd())

    def with_suffix(self, suffix):
        if self.suffix == "":
            return Path(self._path + suffix)
        return Path(self._path[:-len(self.suffix)] + suffix)

    def replace(self, target):
        target = str(target)
        _shellsim_vfs.rename(self._path, target)
        return Path(target)

    def rename(self, target):
        return self.replace(target)

    def unlink(self, missing_ok=False):
        if missing_ok and not self.exists():
            return
        _shellsim_vfs.remove_file(self._path)

    def open(self, mode="r", encoding=None, newline=None):
        return open(self._path, mode, encoding, newline)

    def read_text(self, encoding=None):
        return _shellsim_vfs.read_text(self._path)

    def read_bytes(self):
        return _shellsim_vfs.read_bytes(self._path)

    def write_text(self, data, encoding=None):
        _shellsim_vfs.write_text(self._path, data)
        return len(data)

    def write_bytes(self, data):
        _shellsim_vfs.write_bytes(self._path, data)
        return len(data)


PurePath = Path
PosixPath = Path
