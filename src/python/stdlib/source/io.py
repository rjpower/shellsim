"""Text and binary file protocols backed only by shellsim's modeled VFS."""

import _shellsim_vfs


class _File:
    def __init__(self, path, mode="r"):
        if mode not in ["r", "w", "a", "rb", "wb", "ab"]:
            raise ValueError("only modes r, w, a, rb, wb, and ab are supported")
        self.name = path
        self.mode = mode
        self.closed = False
        self._position = 0
        self._binary = "b" in mode
        self._operation = mode[0]
        self._empty = b"" if self._binary else ""
        if self._operation == "r":
            self._data = self._read_file()
        elif self._operation == "a" and _shellsim_vfs.exists(path):
            self._data = self._read_file()
            self._position = len(self._data)
        else:
            self._data = self._empty
            self._write_file()

    def _read_file(self):
        if self._binary:
            return _shellsim_vfs.read_bytes(self.name)
        return _shellsim_vfs.read_text(self.name)

    def _write_file(self):
        if self._binary:
            _shellsim_vfs.write_bytes(self.name, self._data)
        else:
            _shellsim_vfs.write_text(self.name, self._data)

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
            return self._empty
        end = self._position
        newline = 10 if self._binary else "\n"
        while end < len(self._data) and self._data[end] != newline:
            end += 1
        if end < len(self._data):
            end += 1
        value = self._data[self._position:end]
        self._position = end
        return value

    def readlines(self):
        values = []
        line = self.readline()
        while line != self._empty:
            values.append(line)
            line = self.readline()
        return values

    def write(self, value):
        if self.closed or self._operation == "r":
            raise ValueError("file is not open for writing")
        before = self._data[:self._position]
        after_start = self._position + len(value)
        after = self._data[after_start:]
        self._data = before + value + after
        self._position += len(value)
        self._write_file()
        return len(value)

    def writelines(self, values):
        for value in values:
            self.write(value)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.readline()
        if value == self._empty:
            raise StopIteration
        return value

    def tell(self):
        return self._position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = len(self._data) + offset
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def flush(self):
        if not self.closed and self._operation != "r":
            self._write_file()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False

    def close(self):
        self.closed = True


class TextIOWrapper(_File):
    pass


class BufferedReader(_File):
    pass


class BufferedWriter(_File):
    pass


class _MemoryIO:
    def __init__(self, initial_value):
        self._data = initial_value
        self._position = 0
        self.closed = False

    def getvalue(self):
        return self._data

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

    def write(self, value):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        before = self._data[:self._position]
        after_start = self._position + len(value)
        self._data = before + value + self._data[after_start:]
        self._position += len(value)
        return len(value)

    def tell(self):
        return self._position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = len(self._data) + offset
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False


class StringIO(_MemoryIO):
    def __init__(self, initial_value=""):
        _MemoryIO.__init__(self, initial_value)


class BytesIO(_MemoryIO):
    def __init__(self, initial_bytes=b""):
        _MemoryIO.__init__(self, initial_bytes)


def open(path, mode="r", encoding=None, newline=None):
    if "b" in mode:
        return BufferedReader(str(path), mode)
    return TextIOWrapper(str(path), mode)
