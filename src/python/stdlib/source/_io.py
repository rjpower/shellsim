"""Text and binary file objects backed only by shellsim's modeled VFS.

A file object materializes the whole file in modeled memory when opened and writes it back
through the VFS on each write or flush, so the cost of a file is proportional to its size. The
class hierarchy follows ``io``: ``IOBase`` and its three abstract subclasses, ``FileIO`` for raw
bytes, the ``Buffered*`` classes for binary files, ``TextIOWrapper`` for text, and the in-memory
``BytesIO`` and ``StringIO``. There are no file descriptors: ``fileno()`` raises
``UnsupportedOperation``.
"""

import _shellsim_vfs
from abc import ABCMeta
from _sys import stdin as _stdin, stdout as _stdout, stderr as _stderr

__all__ = [
    "BlockingIOError", "open", "open_code", "IOBase", "RawIOBase", "FileIO", "BytesIO", "StringIO",
    "BufferedIOBase", "BufferedReader", "BufferedWriter", "BufferedRWPair", "BufferedRandom",
    "TextIOBase", "TextIOWrapper", "UnsupportedOperation", "SEEK_SET", "SEEK_CUR", "SEEK_END",
    "DEFAULT_BUFFER_SIZE", "text_encoding", "IncrementalNewlineDecoder", "Reader", "Writer",
]

# Re-exported so ``io`` can import it from here as CPython's ``_io`` provides it.
BlockingIOError = BlockingIOError

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2
DEFAULT_BUFFER_SIZE = 128 * 1024

_ENCODINGS = {
    "utf-8": "utf-8", "utf8": "utf-8", "utf_8": "utf-8", "u8": "utf-8", "locale": "utf-8",
    "ascii": "ascii", "us-ascii": "ascii", "us_ascii": "ascii",
    "latin-1": "latin-1", "latin1": "latin-1", "latin_1": "latin-1", "iso-8859-1": "latin-1",
    "iso8859-1": "latin-1", "iso_8859_1": "latin-1", "l1": "latin-1",
}


class UnsupportedOperation(OSError, ValueError):
    pass


def text_encoding(encoding, stacklevel=2):
    """The name ``open`` resolves a missing encoding to; "locale" is UTF-8 in shellsim."""
    if encoding is None:
        return "locale"
    return encoding


def _normalize_encoding(encoding):
    if encoding is None:
        return "utf-8"
    if not isinstance(encoding, str):
        raise TypeError("open() argument 'encoding' must be str or None, not " + type(encoding).__name__)
    normalized = _ENCODINGS.get(encoding.lower())
    if normalized is None:
        raise LookupError("unknown encoding: " + encoding)
    return normalized


def _parse_mode(mode):
    """Validate an `open()` mode as CPython does and return `(operation, binary, plus)`."""
    if not isinstance(mode, str):
        raise TypeError("open() argument 'mode' must be str, not " + type(mode).__name__)
    seen = ""
    for char in mode:
        if char not in "rwxabt+U" or char in seen:
            raise ValueError("invalid mode: " + repr(mode))
        seen += char
    if "t" in seen and "b" in seen:
        raise ValueError("can't have text and binary mode at once")
    if "U" in seen:
        raise ValueError("invalid mode: " + repr(mode))
    operations = [char for char in seen if char in "rwxa"]
    if len(operations) > 1:
        raise ValueError("must have exactly one of create/read/write/append mode")
    if not operations:
        raise ValueError(
            "Must have exactly one of create/read/write/append mode and at most one plus"
        )
    return operations[0], "b" in seen, "+" in seen


class IOBase(metaclass=ABCMeta):
    """The abstract base of all file objects: closing, iteration and capability queries."""

    closed = False

    def _checkClosed(self, msg=None):
        if self.closed:
            raise ValueError("I/O operation on closed file." if msg is None else msg)

    def _checkReadable(self, msg=None):
        if not self.readable():
            raise UnsupportedOperation("File or stream is not readable." if msg is None else msg)

    def _checkWritable(self, msg=None):
        if not self.writable():
            raise UnsupportedOperation("File or stream is not writable." if msg is None else msg)

    def _checkSeekable(self, msg=None):
        if not self.seekable():
            raise UnsupportedOperation("File or stream is not seekable." if msg is None else msg)

    def seek(self, pos, whence=0):
        raise UnsupportedOperation("seek")

    def tell(self):
        return self.seek(0, 1)

    def truncate(self, pos=None):
        raise UnsupportedOperation("truncate")

    def flush(self):
        self._checkClosed()

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                self.closed = True

    def seekable(self):
        return False

    def readable(self):
        return False

    def writable(self):
        return False

    def fileno(self):
        raise UnsupportedOperation("fileno")

    def isatty(self):
        self._checkClosed()
        return False

    def readline(self, size=-1):
        if size is None:
            size = -1
        result = bytearray()
        while size < 0 or len(result) < size:
            byte = self.read(1)
            if not byte:
                break
            result += byte
            if result.endswith(b"\n"):
                break
        return bytes(result)

    def readlines(self, hint=None):
        if hint is None or hint <= 0:
            return list(self)
        lines = []
        total = 0
        for line in self:
            lines.append(line)
            total += len(line)
            # Reading stops once the lines read so far exceed the hint, as CPython does.
            if total > hint:
                break
        return lines

    def writelines(self, lines):
        self._checkClosed()
        for line in lines:
            self.write(line)

    def __iter__(self):
        self._checkClosed()
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        self._checkClosed()
        return self

    def __exit__(self, *args):
        self.close()


class RawIOBase(IOBase):
    """Unbuffered bytes: subclasses implement ``readinto`` and ``write``."""

    def read(self, size=-1):
        if size is None or size < 0:
            return self.readall()
        buffer = bytearray(size)
        count = self.readinto(buffer)
        if count is None:
            return None
        return bytes(buffer[:count])

    def readall(self):
        result = bytearray()
        while True:
            data = self.read(DEFAULT_BUFFER_SIZE)
            if not data:
                break
            result += data
        return bytes(result)

    def readinto(self, buffer):
        raise UnsupportedOperation("readinto")

    def write(self, data):
        raise UnsupportedOperation("write")


class BufferedIOBase(IOBase):
    """Buffered bytes: ``read`` returns the requested amount unless the file ends."""

    def read(self, size=-1):
        raise UnsupportedOperation("read")

    def read1(self, size=-1):
        return self.read(size)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        return count

    def readinto1(self, buffer):
        return self.readinto(buffer)

    def write(self, data):
        raise UnsupportedOperation("write")

    def detach(self):
        raise UnsupportedOperation("detach")


class TextIOBase(IOBase):
    """Text: ``read`` and ``write`` deal in ``str``."""

    encoding = None
    errors = None
    newlines = None

    def read(self, size=-1):
        raise UnsupportedOperation("read")

    def write(self, s):
        raise UnsupportedOperation("write")

    def detach(self):
        raise UnsupportedOperation("detach")

    def readline(self, size=-1):
        if size is None:
            size = -1
        result = ""
        while size < 0 or len(result) < size:
            char = self.read(1)
            if not char:
                break
            result += char
            if char == "\n":
                break
        return result


def _translate_newlines(text):
    """Universal newlines: ``\\r\\n`` and ``\\r`` become ``\\n``."""
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _Storage:
    """The materialized contents of one VFS file, shared by a file object and its views."""

    def __init__(self, path, operation, binary, plus):
        self.path = path
        self.operation = operation
        self.readable = operation == "r" or plus
        self.writable = operation != "r" or plus
        if operation == "x" and _shellsim_vfs.exists(path):
            raise FileExistsError(17, "File exists", path)
        if operation == "r":
            self.data = _shellsim_vfs.read_bytes(path)
        elif operation == "a" and _shellsim_vfs.exists(path):
            self.data = _shellsim_vfs.read_bytes(path)
        else:
            if _shellsim_vfs.is_dir(path):
                raise IsADirectoryError(21, "Is a directory", path)
            self.data = b""
            _shellsim_vfs.write_bytes(path, b"")
        self.position = len(self.data) if operation == "a" else 0

    def write_back(self):
        _shellsim_vfs.write_bytes(self.path, self.data)

    def read(self, size):
        if size is None or size < 0:
            chunk = self.data[self.position:]
        else:
            chunk = self.data[self.position:self.position + size]
        self.position += len(chunk)
        return chunk

    def write(self, data):
        if self.operation == "a":
            # Append through the VFS so concurrent writers of the same file interleave
            # instead of overwriting each other with a stale snapshot.
            self.data = _shellsim_vfs.read_bytes(self.path) + data
            self.position = _shellsim_vfs.append_bytes(self.path, data)
            return len(data)
        if self.position > len(self.data):
            self.data += b"\x00" * (self.position - len(self.data))
        self.data = self.data[:self.position] + data + self.data[self.position + len(data):]
        self.position += len(data)
        self.write_back()
        return len(data)

    def seek(self, offset, whence):
        if whence == SEEK_SET:
            if offset < 0:
                raise ValueError("negative seek position %r" % (offset,))
            position = offset
        elif whence == SEEK_CUR:
            position = self.position + offset
        elif whence == SEEK_END:
            position = len(self.data) + offset
        else:
            raise ValueError("invalid whence (%r, should be 0, 1 or 2)" % (whence,))
        if position < 0:
            raise OSError(22, "Invalid argument")
        self.position = position
        return position

    def truncate(self, size):
        if size is None:
            size = self.position
        if size < 0:
            raise ValueError("negative size value %r" % (size,))
        if size < len(self.data):
            self.data = self.data[:size]
        elif size > len(self.data):
            self.data += b"\x00" * (size - len(self.data))
        self.write_back()
        return size


_descriptors = [3]


def _next_descriptor():
    """A fresh descriptor number; 0, 1 and 2 belong to the standard streams."""
    _descriptors[0] += 1
    return _descriptors[0] - 1


class FileIO(RawIOBase):
    """Raw bytes of one VFS file."""

    def fileno(self):
        self._checkClosed()
        return self._fd

    def __init__(self, file, mode="r", closefd=True, opener=None):
        if not closefd:
            raise ValueError("Cannot use closefd=False with file name")
        if opener is not None:
            raise UnsupportedOperation("custom openers are not supported")
        operation, binary, plus = _parse_mode(mode.replace("b", "") + "b" if "b" not in mode else mode)
        self.name = file
        self.mode = _binary_mode(operation, plus)
        self._storage = _Storage(str(file), operation, True, plus)
        self._readable = self._storage.readable
        self._writable = self._storage.writable
        self._fd = _next_descriptor()

    def readable(self):
        self._checkClosed()
        return self._readable

    def writable(self):
        self._checkClosed()
        return self._writable

    def seekable(self):
        self._checkClosed()
        return True

    def read(self, size=-1):
        self._checkClosed()
        self._checkReadable()
        return self._storage.read(size)

    def readall(self):
        return self.read(-1)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def write(self, data):
        self._checkClosed()
        self._checkWritable()
        return self._storage.write(bytes(data))

    def seek(self, pos, whence=SEEK_SET):
        self._checkClosed()
        return self._storage.seek(pos, whence)

    def tell(self):
        self._checkClosed()
        return self._storage.position

    def truncate(self, size=None):
        self._checkClosed()
        self._checkWritable()
        return self._storage.truncate(size)

    def __repr__(self):
        if self.closed:
            return "<_io.FileIO [closed]>"
        return "<_io.FileIO name=%r mode=%r closefd=True>" % (self.name, self.mode)


def _binary_mode(operation, plus):
    """The mode CPython's FileIO reports, which spells `w+` as `rb+`."""
    if plus and operation == "w":
        operation = "r"
    return operation + "b" + ("+" if plus else "")


class _Buffered(BufferedIOBase):
    """Shared implementation of the three buffered binary file classes."""

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        if buffer_size <= 0:
            raise ValueError("invalid buffer size")
        self.raw = raw
        self._buffer_size = buffer_size
        # Bytes read ahead from `raw` for `peek` and not yet handed out.
        self._read_ahead = b""

    @property
    def name(self):
        return self.raw.name

    @property
    def mode(self):
        return self.raw.mode

    @property
    def closed(self):
        return self.raw.closed

    def readable(self):
        return self.raw.readable()

    def writable(self):
        return self.raw.writable()

    def seekable(self):
        return self.raw.seekable()

    def _take(self, size):
        """Up to `size` bytes (all when negative) from the read-ahead buffer first."""
        if size is None or size < 0:
            pending = self._read_ahead
            self._read_ahead = b""
            rest = self.raw.readall() if hasattr(self.raw, "readall") else self.raw.read()
            return pending + (rest or b"")
        chunk = self._read_ahead[:size]
        self._read_ahead = self._read_ahead[size:]
        return chunk

    def read(self, size=-1):
        self._checkClosed()
        self._checkReadable()
        if size is None or size < 0:
            return self._take(-1)
        chunk = self._take(size)
        while len(chunk) < size:
            # Short requests fill a whole buffer so `peek` and `read1` see what is left over.
            needed = size - len(chunk)
            more = self.raw.read(max(needed, self._buffer_size))
            if not more:
                break
            chunk += more[:needed]
            self._read_ahead = more[needed:]
        return chunk

    def read1(self, size=-1):
        self._checkClosed()
        self._checkReadable()
        if size is None or size < 0:
            size = self._buffer_size
        if not self._read_ahead:
            self._read_ahead = self.raw.read(max(size, 1)) or b""
        return self._take(size)

    def peek(self, size=0):
        self._checkClosed()
        self._checkReadable()
        if not self._read_ahead:
            self._read_ahead = self.raw.read(self._buffer_size) or b""
        return self._read_ahead

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def write(self, data):
        self._checkClosed()
        self._checkWritable()
        self._discard_read_ahead()
        return self.raw.write(data)

    def _discard_read_ahead(self):
        """Give unread read-ahead bytes back to a seekable raw before the position changes."""
        if self._read_ahead:
            if self.raw.seekable():
                self.raw.seek(-len(self._read_ahead), SEEK_CUR)
            self._read_ahead = b""

    def seek(self, pos, whence=SEEK_SET):
        self._checkClosed()
        if whence == SEEK_CUR:
            pos -= len(self._read_ahead)
        self._read_ahead = b""
        return self.raw.seek(pos, whence)

    def tell(self):
        self._checkClosed()
        return self.raw.tell() - len(self._read_ahead)

    def truncate(self, size=None):
        self._checkClosed()
        if size is None:
            size = self.tell()
        self._discard_read_ahead()
        return self.raw.truncate(size)

    def flush(self):
        self._checkClosed()

    def close(self):
        if not self.raw.closed:
            self.raw.close()

    def detach(self):
        raw = self.raw
        self.raw = None
        return raw

    def fileno(self):
        return self.raw.fileno()

    def isatty(self):
        return self.raw.isatty()

    def __repr__(self):
        name = getattr(self.raw, "name", None)
        if name is None:
            return "<%s>" % type(self).__name__
        return "<%s name=%r>" % (type(self).__name__, name)


class BufferedReader(_Buffered):
    pass


class BufferedWriter(_Buffered):
    pass


class BufferedRandom(_Buffered):
    pass


class BufferedRWPair(BufferedIOBase):
    """A reader and a writer joined into one object."""

    def __init__(self, reader, writer, buffer_size=DEFAULT_BUFFER_SIZE):
        if not reader.readable():
            raise UnsupportedOperation('"reader" argument must be readable.')
        if not writer.writable():
            raise UnsupportedOperation('"writer" argument must be writable.')
        self.reader = reader
        self.writer = writer

    def read(self, size=-1):
        return self.reader.read(size)

    def read1(self, size=-1):
        return self.reader.read1(size)

    def readinto(self, buffer):
        return self.reader.readinto(buffer)

    def peek(self, size=0):
        return self.reader.peek(size)

    def write(self, data):
        return self.writer.write(data)

    def flush(self):
        return self.writer.flush()

    def readable(self):
        return self.reader.readable()

    def writable(self):
        return self.writer.writable()

    def close(self):
        try:
            self.writer.close()
        finally:
            self.reader.close()

    def isatty(self):
        return self.reader.isatty() or self.writer.isatty()

    @property
    def closed(self):
        return self.writer.closed


class IncrementalNewlineDecoder:
    """Translate ``\\r\\n`` and ``\\r`` to ``\\n`` across chunk boundaries and record which
    newline kinds were seen."""

    def __init__(self, decoder, translate, errors="strict"):
        self.decoder = decoder
        self.translate = translate
        self.errors = errors
        self.pendingcr = False
        self.seennl = 0

    _LF = 1
    _CR = 2
    _CRLF = 4

    def decode(self, input, final=False):
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final=final)
        if self.pendingcr and (output or final):
            output = "\r" + output
            self.pendingcr = False
        if output.endswith("\r") and not final:
            output = output[:-1]
            self.pendingcr = True
        crlf = output.count("\r\n")
        cr = output.count("\r") - crlf
        lf = output.count("\n") - crlf
        self.seennl |= (lf and self._LF) | (cr and self._CR) | (crlf and self._CRLF)
        if self.translate:
            output = _translate_newlines(output)
        return output

    def getstate(self):
        if self.decoder is None:
            buffer = b""
            flag = 0
        else:
            buffer, flag = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return buffer, flag

    def setstate(self, state):
        buffer, flag = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buffer, flag >> 1))

    def reset(self):
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()

    @property
    def newlines(self):
        return (None, "\n", "\r", ("\r", "\n"), "\r\n", ("\n", "\r\n"), ("\r", "\r\n"),
                ("\r", "\n", "\r\n"))[self.seennl]


class TextIOWrapper(TextIOBase):
    """Text over a binary file: decodes on read, encodes on write, translates newlines."""

    def __init__(self, buffer, encoding=None, errors=None, newline=None, line_buffering=False,
                 write_through=False):
        if newline is not None and not isinstance(newline, str):
            raise TypeError("illegal newline type: " + type(newline).__name__)
        if newline not in (None, "", "\n", "\r", "\r\n"):
            raise ValueError("illegal newline value: " + repr(newline))
        self.buffer = buffer
        self._encoding = _normalize_encoding(encoding)
        self._errors = "strict" if errors is None else errors
        self._readuniversal = not newline
        self._readtranslate = newline is None
        self._writetranslate = newline != ""
        self._writenl = newline or "\n"
        self.line_buffering = line_buffering
        self.write_through = write_through
        self._seen_newlines = 0
        self._decoded = None
        self._decoded_from = None
        self._stream_cache = None
        self._position = 0

    @property
    def encoding(self):
        return self._encoding

    @property
    def errors(self):
        return self._errors

    @property
    def name(self):
        return self.buffer.name

    @property
    def mode(self):
        mode = getattr(self, "_mode_text", None)
        if mode is not None:
            return mode
        storage = self._storage()
        if storage is None:
            raise AttributeError("mode")
        return storage.operation + ("+" if storage.readable and storage.writable else "")

    def _storage(self):
        """The VFS file behind an `open()` buffer, or None over an arbitrary binary buffer."""
        raw = getattr(self.buffer, "raw", None)
        return getattr(raw, "_storage", None)

    def _bytes(self):
        """The buffer's complete current contents."""
        storage = self._storage()
        if storage is not None:
            return storage.data
        buffer = self.buffer
        if buffer.seekable():
            position = buffer.tell()
            buffer.seek(0)
            data = buffer.read()
            buffer.seek(position)
            return data
        if self._stream_cache is None:
            self._stream_cache = buffer.read()
        return self._stream_cache

    def _store(self, data):
        """Replace the buffer's contents with `data`."""
        storage = self._storage()
        if storage is not None:
            storage.data = data
            storage.position = len(data)
            storage.write_back()
            return
        buffer = self.buffer
        buffer.seek(0)
        buffer.truncate(0)
        buffer.write(data)

    @property
    def closed(self):
        return self.buffer.closed

    @property
    def newlines(self):
        return _newline_kinds(self._seen_newlines)

    def readable(self):
        return self.buffer.readable()

    def writable(self):
        return self.buffer.writable()

    def seekable(self):
        return self.buffer.seekable()

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def _text(self):
        """The decoded contents, with the character position kept in step with the bytes."""
        data = self._bytes()
        if self._decoded is None or self._decoded_from != data:
            text = data.decode(self._encoding, self._errors)
            crlf = text.count("\r\n")
            cr = text.count("\r") - crlf
            lf = text.count("\n") - crlf
            self._seen_newlines |= (lf and 1) | (cr and 2) | (crlf and 4)
            if self._readtranslate:
                text = _translate_newlines(text)
            self._decoded = text
            self._decoded_from = data
        return self._decoded

    def read(self, size=-1):
        self._checkClosed()
        self._checkReadable()
        text = self._text()
        if size is None or size < 0:
            chunk = text[self._position:]
        else:
            chunk = text[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def readline(self, size=-1):
        self._checkClosed()
        self._checkReadable()
        if size is None:
            size = -1
        text = self._text()
        start = self._position
        if start >= len(text):
            return ""
        if self._readuniversal:
            end = len(text)
            for marker in ("\n", "\r"):
                found = text.find(marker, start)
                if found != -1 and found < end:
                    end = found
            if end < len(text):
                end += 2 if text[end:end + 2] == "\r\n" else 1
        else:
            found = text.find(self._writenl, start)
            end = len(text) if found == -1 else found + len(self._writenl)
        if size >= 0:
            end = min(end, start + size)
        self._position = end
        return text[start:end]

    def write(self, s):
        self._checkClosed()
        self._checkWritable()
        if not isinstance(s, str):
            raise TypeError("write() argument must be str, not " + type(s).__name__)
        if self._writetranslate and self._writenl != "\n":
            s = s.replace("\n", self._writenl)
        storage = self._storage()
        if storage is None and not self.buffer.seekable():
            # A pipe-like buffer only appends; nothing read back needs to stay in step.
            self.buffer.write(s.encode(self._encoding, self._errors))
            self._position += len(s)
            return len(s)
        if storage is not None and storage.operation == "a":
            # Append through the raw file so that another process appending to the same
            # path interleaves with this one instead of being overwritten by a snapshot.
            self.buffer.write(s.encode(self._encoding, self._errors))
            self._position = len(self._text())
            return len(s)
        text = self._text()
        before = text[:self._position]
        after = text[self._position + len(s):]
        if self._position > len(text):
            before = text + "\x00" * (self._position - len(text))
        new_text = before + s + after
        data = new_text.encode(self._encoding, self._errors)
        self._store(data)
        self._decoded = new_text
        self._decoded_from = data
        self._position += len(s)
        return len(s)

    def _byte_offset(self, position):
        """Encoded length of the raw text behind the first `position` decoded characters.

        Universal-newline translation only collapses "\r\n" pairs, so the raw prefix is the
        translated prefix plus one extra character per collapsed pair.
        """
        raw = self._bytes().decode(self._encoding, self._errors)
        if not self._readtranslate:
            return len(raw[:position].encode(self._encoding, self._errors))
        consumed = 0
        index = 0
        while consumed < position and index < len(raw):
            if raw[index] == "\r" and raw[index + 1:index + 2] == "\n":
                index += 2
            else:
                index += 1
            consumed += 1
        return len(raw[:index].encode(self._encoding, self._errors))

    def tell(self):
        """The position as a byte offset into the underlying buffer, as CPython's cookie is."""
        self._checkClosed()
        return self._byte_offset(self._position)

    def seek(self, cookie, whence=SEEK_SET):
        self._checkClosed()
        if whence == SEEK_CUR:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero cur-relative seeks")
            return self.tell()
        if whence == SEEK_END:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero end-relative seeks")
            self._position = len(self._text())
            return self.tell()
        if whence != SEEK_SET:
            raise ValueError("invalid whence (%r, should be 0, 1 or 2)" % (whence,))
        if cookie < 0:
            raise ValueError("negative seek position %r" % (cookie,))
        prefix = self._bytes()[:cookie].decode(self._encoding, self._errors)
        if self._readtranslate:
            prefix = _translate_newlines(prefix)
        self._position = len(prefix)
        return cookie

    def truncate(self, pos=None):
        self._checkClosed()
        self._checkWritable()
        if pos is None:
            pos = self._position
        text = self._text()[:pos]
        data = text.encode(self._encoding, self._errors)
        self._store(data)
        self._decoded = text
        self._decoded_from = data
        return pos

    def flush(self):
        self._checkClosed()
        self.buffer.flush()

    def close(self):
        if not self.closed:
            self.buffer.close()

    def detach(self):
        buffer = self.buffer
        self.buffer = None
        return buffer

    def reconfigure(self, *, encoding=None, errors=None, newline=None, line_buffering=None,
                    write_through=None):
        if encoding is not None:
            self._encoding = _normalize_encoding(encoding)
        if errors is not None:
            self._errors = errors
        if line_buffering is not None:
            self.line_buffering = line_buffering
        if write_through is not None:
            self.write_through = write_through
        self._decoded = None

    def __repr__(self):
        name = getattr(self.buffer, "name", None)
        if name is None:
            return "<_io.TextIOWrapper encoding=%r>" % self._encoding
        return "<_io.TextIOWrapper name=%r mode=%r encoding=%r>" % (name, self.mode, self._encoding)


def _newline_kinds(seen):
    return (None, "\n", "\r", ("\r", "\n"), "\r\n", ("\n", "\r\n"), ("\r", "\r\n"),
            ("\r", "\n", "\r\n"))[seen]


class _BufferView:
    """What `BytesIO.getbuffer()` returns: a read view with memoryview's common surface."""

    def __init__(self, data):
        self._data = bytes(data)

    nbytes = property(lambda self: len(self._data))
    readonly = property(lambda self: True)
    itemsize = property(lambda self: 1)
    ndim = property(lambda self: 1)
    shape = property(lambda self: (len(self._data),))
    format = property(lambda self: "B")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _BufferView(self._data[index])
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def __bytes__(self):
        return self._data

    def __eq__(self, other):
        return self._data == bytes(other)

    def tobytes(self):
        return self._data

    def tolist(self):
        return list(self._data)

    def hex(self):
        return self._data.hex()

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class BytesIO(BufferedIOBase):
    """An in-memory binary file."""

    def __init__(self, initial_bytes=b""):
        self._data = bytes(initial_bytes)
        self._position = 0
        self.closed = False

    def getvalue(self):
        self._checkClosed()
        return self._data

    def getbuffer(self):
        self._checkClosed()
        return _BufferView(self._data)

    def readable(self):
        self._checkClosed()
        return True

    def writable(self):
        self._checkClosed()
        return True

    def seekable(self):
        self._checkClosed()
        return True

    def read(self, size=-1):
        self._checkClosed()
        if size is None or size < 0:
            chunk = self._data[self._position:]
        else:
            chunk = self._data[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def read1(self, size=-1):
        return self.read(size)

    def readline(self, size=-1):
        self._checkClosed()
        if size is None:
            size = -1
        end = self._data.find(b"\n", self._position)
        end = len(self._data) if end < 0 else end + 1
        if size >= 0:
            end = min(end, self._position + size)
        chunk = self._data[self._position:end]
        self._position = max(self._position, end)
        return chunk

    def write(self, data):
        self._checkClosed()
        data = bytes(data)
        if self._position > len(self._data):
            self._data += b"\x00" * (self._position - len(self._data))
        self._data = self._data[:self._position] + data + self._data[self._position + len(data):]
        self._position += len(data)
        return len(data)

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, pos, whence=SEEK_SET):
        self._checkClosed()
        if whence == SEEK_SET:
            if pos < 0:
                raise ValueError("negative seek value %r" % (pos,))
            position = pos
        elif whence == SEEK_CUR:
            position = max(0, self._position + pos)
        elif whence == SEEK_END:
            position = max(0, len(self._data) + pos)
        else:
            raise ValueError("invalid whence (%r, should be 0, 1 or 2)" % (whence,))
        self._position = position
        return position

    def truncate(self, size=None):
        self._checkClosed()
        if size is None:
            size = self._position
        if size < 0:
            raise ValueError("negative size value %r" % (size,))
        self._data = self._data[:size]
        return size

    def flush(self):
        self._checkClosed()

    def __repr__(self):
        return "<_io.BytesIO object at 0x%x>" % id(self)


class StringIO(TextIOBase):
    """An in-memory text file with CPython's newline handling."""

    def __init__(self, initial_value="", newline="\n"):
        if newline is not None and not isinstance(newline, str):
            raise TypeError("newline must be str or None, not " + type(newline).__name__)
        if newline not in (None, "", "\n", "\r", "\r\n"):
            raise ValueError("illegal newline value: " + repr(newline))
        if initial_value is None:
            initial_value = ""
        if not isinstance(initial_value, str):
            raise TypeError("initial_value must be str or None, not " + type(initial_value).__name__)
        self._readtranslate = newline is None
        self._writenl = newline or "\n"
        self._writetranslate = newline is not None and newline != ""
        self._seen_newlines = 0
        self._data = ""
        self._position = 0
        self.closed = False
        if initial_value:
            self.write(initial_value)
            self._position = 0

    @property
    def encoding(self):
        return None

    @property
    def errors(self):
        return None

    @property
    def line_buffering(self):
        return False

    @property
    def newlines(self):
        """The newline kinds seen so far when translating (`newline=None`), else None."""
        if not self._readtranslate:
            return None
        return _newline_kinds(self._seen_newlines)

    def getvalue(self):
        self._checkClosed()
        return self._data

    def readable(self):
        self._checkClosed()
        return True

    def writable(self):
        self._checkClosed()
        return True

    def seekable(self):
        self._checkClosed()
        return True

    def read(self, size=-1):
        self._checkClosed()
        if size is None or size < 0:
            chunk = self._data[self._position:]
        else:
            chunk = self._data[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def readline(self, size=-1):
        self._checkClosed()
        if size is None:
            size = -1
        text = self._data
        start = self._position
        if self._readtranslate or self._writenl == "\n":
            end = text.find("\n", start)
            end = len(text) if end < 0 else end + 1
        else:
            end = text.find(self._writenl, start)
            end = len(text) if end < 0 else end + len(self._writenl)
        if size >= 0:
            end = min(end, start + size)
        self._position = max(start, end)
        return text[start:end]

    def write(self, s):
        self._checkClosed()
        if not isinstance(s, str):
            raise TypeError("string argument expected, got '%s'" % type(s).__name__)
        length = len(s)
        if self._readtranslate:
            crlf = s.count("\r\n")
            cr = s.count("\r") - crlf
            lf = s.count("\n") - crlf
            self._seen_newlines |= (lf and 1) | (cr and 2) | (crlf and 4)
            s = _translate_newlines(s)
        elif self._writetranslate and self._writenl != "\n":
            s = s.replace("\n", self._writenl)
        if self._position > len(self._data):
            self._data += "\x00" * (self._position - len(self._data))
        self._data = self._data[:self._position] + s + self._data[self._position + len(s):]
        self._position += len(s)
        return length

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, pos, whence=SEEK_SET):
        self._checkClosed()
        if whence == SEEK_SET:
            if pos < 0:
                raise ValueError("Negative seek position %r" % (pos,))
            position = pos
        elif whence == SEEK_CUR:
            if pos != 0:
                raise UnsupportedOperation("Can't do nonzero cur-relative seeks")
            position = self._position
        elif whence == SEEK_END:
            if pos != 0:
                raise UnsupportedOperation("Can't do nonzero end-relative seeks")
            position = len(self._data)
        else:
            raise ValueError("Invalid whence (%r, should be 0, 1 or 2)" % (whence,))
        self._position = position
        return position

    def truncate(self, pos=None):
        self._checkClosed()
        if pos is None:
            pos = self._position
        if pos < 0:
            raise ValueError("Negative size value %r" % (pos,))
        self._data = self._data[:pos]
        return pos

    def flush(self):
        self._checkClosed()

    def detach(self):
        raise UnsupportedOperation("detach")

    def __repr__(self):
        return "<_io.StringIO object at 0x%x>" % id(self)


class Reader:
    """Protocol-style base for objects with ``read``."""

    def read(self, size=-1, /):
        raise NotImplementedError


class Writer:
    """Protocol-style base for objects with ``write``."""

    def write(self, data, /):
        raise NotImplementedError


def open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True,
         opener=None):
    """Open a VFS path. Integer file descriptors are not supported."""
    if isinstance(file, int):
        raise UnsupportedOperation("file descriptors are not supported")
    if not isinstance(file, (str, bytes)):
        fspath = getattr(type(file), "__fspath__", None)
        if fspath is None:
            raise TypeError("invalid file: %r" % (file,))
        file = fspath(file)
    if isinstance(file, bytes):
        file = file.decode("utf-8")
    operation, binary, plus = _parse_mode(mode)
    if not isinstance(buffering, int):
        raise TypeError("'%s' object cannot be interpreted as an integer" % type(buffering).__name__)
    if binary:
        if encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")
        if errors is not None:
            raise ValueError("binary mode doesn't take an errors argument")
        if newline is not None:
            raise ValueError("binary mode doesn't take a newline argument")
    elif buffering == 0:
        raise ValueError("can't have unbuffered text I/O")
    raw = FileIO(file, operation + "b" + ("+" if plus else ""), closefd, opener)
    if binary and buffering == 0:
        return raw
    if plus:
        buffer = BufferedRandom(raw)
    elif operation == "r":
        buffer = BufferedReader(raw)
    else:
        buffer = BufferedWriter(raw)
    if binary:
        return buffer
    text = TextIOWrapper(buffer, encoding, errors, newline, line_buffering=buffering == 1)
    text._mode_text = mode
    return text


def open_code(path):
    if not isinstance(path, str):
        raise TypeError("'path' must be 'str', not '%s'" % type(path).__name__)
    return open(path, "rb")


# The standard streams are native objects; register their type so `isinstance(sys.stdout,
# io.TextIOBase)` holds as it does in CPython.
TextIOBase.register(type(_stdout))
