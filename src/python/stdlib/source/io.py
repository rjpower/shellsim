"""Text and binary file protocols backed only by shellsim's modeled VFS."""

import _shellsim_vfs


def _parse_mode(mode):
    """Validate an `open()` mode as CPython does and return `(operation, binary, plus)`."""
    if not isinstance(mode, str):
        raise TypeError("open() argument 'mode' must be str, not " + type(mode).__name__)
    seen = ""
    for char in mode:
        if char not in "rwxabt+" or char in seen:
            raise ValueError("invalid mode: " + repr(mode))
        seen += char
    if "t" in seen and "b" in seen:
        raise ValueError("can't have text and binary mode at once")
    operations = [char for char in seen if char in "rwxa"]
    if len(operations) > 1:
        raise ValueError("must have exactly one of create/read/write/append mode")
    if not operations:
        raise ValueError(
            "Must have exactly one of create/read/write/append mode and at most one plus"
        )
    return operations[0], "b" in seen, "+" in seen


def _binary_mode(operation, plus):
    """The mode CPython's FileIO reports, which spells `w+` as `rb+`."""
    if plus and operation == "w":
        operation = "r"
    return operation + "b" + ("+" if plus else "")


class _File:
    def __init__(self, path, mode="r"):
        operation, binary, plus = _parse_mode(mode)
        self.name = path
        self.mode = _binary_mode(operation, plus) if binary else mode
        self.closed = False
        self._position = 0
        self._binary = binary
        self._operation = operation
        self._readable = operation == "r" or plus
        self._writable = operation != "r" or plus
        self._empty = b"" if self._binary else ""
        if operation == "x" and _shellsim_vfs.exists(path):
            raise FileExistsError("[Errno 17] File exists: " + repr(path))
        if operation == "r":
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
        if not self._readable:
            raise ValueError("file is not open for reading")
        if size < 0:
            value = self._data[self._position:]
            self._position = len(self._data)
        else:
            value = self._data[self._position:self._position + size]
            self._position += len(value)
        return value

    def readline(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if not self._readable:
            raise ValueError("file is not open for reading")
        if self._position >= len(self._data):
            return self._empty
        newline = b"\n" if self._binary else "\n"
        end = self._data.find(newline, self._position)
        end = len(self._data) if end < 0 else end + 1
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
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if not self._writable:
            raise ValueError("file is not open for writing")
        if self._operation == "a":
            if self._binary:
                self._position = _shellsim_vfs.append_bytes(self.name, value)
            else:
                self._position = _shellsim_vfs.append_text(self.name, value)
            self._data += value
            return len(value)
        if self._position > len(self._data):
            padding = b"\x00" if self._binary else "\x00"
            self._data += padding * (self._position - len(self._data))
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
        if not self.closed and self._writable:
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


class BufferedRandom(_File):
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

    def readline(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        newline = b"\n" if isinstance(self._data, bytes) else "\n"
        end = self._data.find(newline, self._position)
        end = len(self._data) if end < 0 else end + 1
        if size is not None and size >= 0:
            end = min(end, self._position + size)
        value = self._data[self._position:end]
        self._position = max(self._position, end)
        return value

    def readlines(self, hint=-1):
        values = []
        total = 0
        line = self.readline()
        while line:
            values.append(line)
            total += len(line)
            if hint is not None and 0 < hint < total:
                break
            line = self.readline()
        return values

    def __iter__(self):
        return self

    def __next__(self):
        value = self.readline()
        if not value:
            raise StopIteration
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
    operation, binary, plus = _parse_mode(mode)
    if not binary:
        return TextIOWrapper(str(path), mode)
    if encoding is not None:
        raise ValueError("binary mode doesn't take an encoding argument")
    if newline is not None:
        raise ValueError("binary mode doesn't take a newline argument")
    if plus:
        return BufferedRandom(str(path), mode)
    if operation == "r":
        return BufferedReader(str(path), mode)
    return BufferedWriter(str(path), mode)
