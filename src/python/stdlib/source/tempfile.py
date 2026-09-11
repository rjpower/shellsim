"""Deterministic temporary paths confined to shellsim's modeled VFS."""

import _shellsim_vfs
import uuid


def _next_path(directory, prefix, suffix):
    if directory is None:
        directory = "/tmp"
    if not directory.endswith("/"):
        directory += "/"
    return directory + prefix + uuid.uuid4().hex + suffix


def mkdtemp(suffix="", prefix="tmp", dir=None):
    path = _next_path(dir, prefix, suffix)
    _shellsim_vfs.mkdir(path, True, False)
    return path


class TemporaryDirectory:
    def __init__(self, suffix="", prefix="tmp", dir=None):
        self.name = mkdtemp(suffix, prefix, dir)

    def cleanup(self):
        if _shellsim_vfs.exists(self.name):
            _shellsim_vfs.remove_tree(self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, kind, value, traceback):
        self.cleanup()
        return False


class _NamedTemporaryFile:
    def __init__(self, path, mode, delete):
        self.name = path
        self.delete = delete
        self._file = open(path, mode)

    def write(self, value):
        return self._file.write(value)

    def read(self, size=-1):
        return self._file.read(size)

    def flush(self):
        return self._file.flush()

    def close(self):
        self._file.close()
        if self.delete and _shellsim_vfs.exists(self.name):
            _shellsim_vfs.remove_file(self.name)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False


def NamedTemporaryFile(mode="w+b", suffix="", prefix="tmp", dir=None, delete=True):
    if mode == "w+b":
        mode = "wb"
    path = _next_path(dir, prefix, suffix)
    return _NamedTemporaryFile(path, mode, delete)
