"""Text file protocol backed only by shellsim's modeled VFS."""

import _shellsim_vfs


class TextIOWrapper:
    def __init__(self, path, mode="r"):
        if mode not in ["r", "w", "a"]:
            raise ValueError("only text modes r, w, and a are supported")
        self.name = path
        self.mode = mode
        self.closed = False
        self._position = 0
        if mode == "r":
            self._data = _shellsim_vfs.read_text(path)
        elif mode == "a" and _shellsim_vfs.exists(path):
            self._data = _shellsim_vfs.read_text(path)
            self._position = len(self._data)
        else:
            self._data = ""
            _shellsim_vfs.write_text(path, "")

    def read(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if size < 0:
            value = self._data[self._position:]
            self._position = len(self._data)
        else:
            value = self._data[self._position:self._position + size]
            self._position += len(value)
        return value

    def readline(self):
        if self._position >= len(self._data):
            return ""
        end = self._position
        while end < len(self._data) and self._data[end] != "\n":
            end += 1
        if end < len(self._data):
            end += 1
        value = self._data[self._position:end]
        self._position = end
        return value

    def readlines(self):
        values = []
        line = self.readline()
        while line != "":
            values.append(line)
            line = self.readline()
        return values

    def write(self, value):
        if self.closed or self.mode == "r":
            raise ValueError("file is not open for writing")
        before = self._data[:self._position]
        after_start = self._position + len(value)
        after = self._data[after_start:]
        self._data = before + value + after
        self._position += len(value)
        _shellsim_vfs.write_text(self.name, self._data)
        return len(value)

    def writelines(self, values):
        for value in values:
            self.write(value)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.readline()
        if value == "":
            raise StopIteration
        return value

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False

    def close(self):
        self.closed = True


def open(path, mode="r", encoding=None, newline=None):
    return TextIOWrapper(str(path), mode)
