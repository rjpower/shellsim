"""Small Path facade over shellsim's modeled VFS."""

import _shellsim_vfs


def _join(left, right):
    if right.startswith("/"):
        return right
    if left == "" or left.endswith("/"):
        return left + right
    return left + "/" + right


class Path:
    def __init__(self, value="."):
        self._path = str(value)

    def __str__(self):
        return self._path

    def __repr__(self):
        return "PosixPath('" + self._path + "')"

    def __truediv__(self, other):
        return Path(_join(self._path, str(other)))

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

    def open(self, mode="r", encoding=None, newline=None):
        return open(self._path, mode, encoding, newline)

    def read_text(self, encoding=None):
        return _shellsim_vfs.read_text(self._path)

    def read_bytes(self):
        return _shellsim_vfs.read_text(self._path)

    def write_text(self, data, encoding=None):
        _shellsim_vfs.write_text(self._path, data)
        return len(data)

    def write_bytes(self, data):
        _shellsim_vfs.write_text(self._path, data)
        return len(data)


PurePath = Path
PosixPath = Path
