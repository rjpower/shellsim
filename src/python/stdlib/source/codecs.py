"""Codec registry, incremental codecs and stream wrappers.

The text codecs are the ones ``str.encode`` understands: UTF-8, ASCII and Latin-1, plus the
``rot13`` text transform. ``lookup`` raises ``LookupError`` for anything else, so a program
finds out explicitly that an encoding is unavailable.
"""

import io as _io

__all__ = [
    "register", "lookup", "open", "EncodedFile", "BOM", "BOM_BE", "BOM_LE", "BOM32_BE",
    "BOM32_LE", "BOM64_BE", "BOM64_LE", "BOM_UTF8", "BOM_UTF16", "BOM_UTF16_LE", "BOM_UTF16_BE",
    "BOM_UTF32", "BOM_UTF32_LE", "BOM_UTF32_BE", "CodecInfo", "Codec", "IncrementalEncoder",
    "IncrementalDecoder", "StreamReader", "StreamWriter", "StreamReaderWriter", "StreamRecoder",
    "getencoder", "getdecoder", "getincrementalencoder", "getincrementaldecoder", "getreader",
    "getwriter", "encode", "decode", "iterencode", "iterdecode", "strict_errors",
    "ignore_errors", "replace_errors", "xmlcharrefreplace_errors", "backslashreplace_errors",
    "namereplace_errors", "register_error", "lookup_error", "unregister",
]

BOM_UTF8 = b"\xef\xbb\xbf"
BOM_LE = BOM_UTF16_LE = b"\xff\xfe"
BOM_BE = BOM_UTF16_BE = b"\xfe\xff"
BOM_UTF32_LE = b"\xff\xfe\x00\x00"
BOM_UTF32_BE = b"\x00\x00\xfe\xff"
BOM = BOM_UTF16 = BOM_UTF16_LE
BOM_UTF32 = BOM_UTF32_LE
BOM32_LE = BOM_UTF16_LE
BOM32_BE = BOM_UTF16_BE
BOM64_LE = BOM_UTF32_LE
BOM64_BE = BOM_UTF32_BE


class CodecInfo(tuple):
    """The functions of one codec, as returned by ``lookup``."""

    _is_text_encoding = True

    def __new__(cls, encode, decode, streamreader=None, streamwriter=None,
                incrementalencoder=None, incrementaldecoder=None, name=None,
                *, _is_text_encoding=None):
        self = tuple.__new__(cls, (encode, decode, streamreader, streamwriter))
        self.name = name
        self.encode = encode
        self.decode = decode
        self.incrementalencoder = incrementalencoder
        self.incrementaldecoder = incrementaldecoder
        self.streamwriter = streamwriter
        self.streamreader = streamreader
        if _is_text_encoding is not None:
            self._is_text_encoding = _is_text_encoding
        return self

    def __repr__(self):
        return "<codecs.CodecInfo object for encoding %s>" % self.name


class Codec:
    def encode(self, input, errors="strict"):
        raise NotImplementedError

    def decode(self, input, errors="strict"):
        raise NotImplementedError


class IncrementalEncoder:
    def __init__(self, errors="strict"):
        self.errors = errors
        self.buffer = ""

    def encode(self, input, final=False):
        raise NotImplementedError

    def reset(self):
        pass

    def getstate(self):
        return 0

    def setstate(self, state):
        pass


class BufferedIncrementalEncoder(IncrementalEncoder):
    def __init__(self, errors="strict"):
        IncrementalEncoder.__init__(self, errors)
        self.buffer = ""

    def _buffer_encode(self, input, errors, final):
        raise NotImplementedError

    def encode(self, input, final=False):
        data = self.buffer + input
        result, consumed = self._buffer_encode(data, self.errors, final)
        self.buffer = data[consumed:]
        return result

    def reset(self):
        IncrementalEncoder.reset(self)
        self.buffer = ""

    def getstate(self):
        return self.buffer or 0

    def setstate(self, state):
        self.buffer = state or ""


class IncrementalDecoder:
    def __init__(self, errors="strict"):
        self.errors = errors

    def decode(self, input, final=False):
        raise NotImplementedError

    def reset(self):
        pass

    def getstate(self):
        return (b"", 0)

    def setstate(self, state):
        pass


class BufferedIncrementalDecoder(IncrementalDecoder):
    def __init__(self, errors="strict"):
        IncrementalDecoder.__init__(self, errors)
        self.buffer = b""

    def _buffer_decode(self, input, errors, final):
        raise NotImplementedError

    def decode(self, input, final=False):
        data = self.buffer + bytes(input)
        result, consumed = self._buffer_decode(data, self.errors, final)
        self.buffer = data[consumed:]
        return result

    def reset(self):
        IncrementalDecoder.reset(self)
        self.buffer = b""

    def getstate(self):
        return (self.buffer, 0)

    def setstate(self, state):
        self.buffer = state[0]


class StreamWriter(Codec):
    def __init__(self, stream, errors="strict"):
        self.stream = stream
        self.errors = errors

    def write(self, object):
        data, consumed = self.encode(object, self.errors)
        self.stream.write(data)

    def writelines(self, list):
        self.write("".join(list))

    def reset(self):
        pass

    def seek(self, offset, whence=0):
        self.stream.seek(offset, whence)
        if whence == 0 and offset == 0:
            self.reset()

    def __getattr__(self, name, getattr=getattr):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()


class StreamReader(Codec):
    charbuffertype = str

    def __init__(self, stream, errors="strict"):
        self.stream = stream
        self.errors = errors
        self.bytebuffer = b""
        self._empty_charbuffer = self.charbuffertype()
        self.charbuffer = self._empty_charbuffer
        self.linebuffer = None

    def decode(self, input, errors="strict"):
        raise NotImplementedError

    def read(self, size=-1, chars=-1, firstline=False):
        if self.linebuffer:
            self.charbuffer = self._empty_charbuffer.join(self.linebuffer)
            self.linebuffer = None
        if chars < 0:
            chars = size
        while True:
            if chars >= 0 and len(self.charbuffer) >= chars:
                break
            newdata = self.stream.read() if size < 0 else self.stream.read(size)
            data = self.bytebuffer + newdata
            if not data:
                break
            try:
                newchars, decodedbytes = self.decode(data, self.errors)
            except UnicodeDecodeError as exc:
                if firstline:
                    newchars, decodedbytes = self.decode(data[:exc.start], self.errors)
                    lines = newchars.splitlines(keepends=True)
                    if len(lines) <= 1:
                        raise
                else:
                    raise
            self.bytebuffer = data[decodedbytes:]
            self.charbuffer += newchars
            if not newdata:
                break
        if chars < 0:
            result = self.charbuffer
            self.charbuffer = self._empty_charbuffer
        else:
            result = self.charbuffer[:chars]
            self.charbuffer = self.charbuffer[chars:]
        return result

    def readline(self, size=None, keepends=True):
        if self.linebuffer:
            line = self.linebuffer[0]
            del self.linebuffer[0]
            if len(self.linebuffer) == 1:
                self.charbuffer = self.linebuffer[0]
                self.linebuffer = None
            if not keepends:
                line = line.splitlines(keepends=False)[0]
            return line
        readsize = size or 72
        line = self._empty_charbuffer
        while True:
            data = self.read(readsize, firstline=True)
            if data:
                if isinstance(data, str) and data.endswith("\r"):
                    data += self.read(size=1, chars=1)
            line += data
            lines = line.splitlines(keepends=True)
            if lines:
                if len(lines) > 1:
                    line = lines[0]
                    del lines[0]
                    if len(lines) > 1:
                        lines[-1] += self.charbuffer
                        self.linebuffer = lines
                        self.charbuffer = None
                    else:
                        self.charbuffer = lines[0] + self.charbuffer
                    if not keepends:
                        line = line.splitlines(keepends=False)[0]
                    break
                line0withend = lines[0]
                line0withoutend = lines[0].splitlines(keepends=False)[0]
                if line0withend != line0withoutend:
                    self.charbuffer = self._empty_charbuffer.join(lines[1:]) + self.charbuffer
                    line = line0withend if keepends else line0withoutend
                    break
            if not data or size is not None:
                if line and not keepends:
                    line = line.splitlines(keepends=False)[0]
                break
            if readsize < 8000:
                readsize *= 2
        return line

    def readlines(self, sizehint=None, keepends=True):
        data = self.read()
        return data.splitlines(keepends)

    def reset(self):
        self.bytebuffer = b""
        self.charbuffer = self._empty_charbuffer
        self.linebuffer = None

    def seek(self, offset, whence=0):
        self.stream.seek(offset, whence)
        self.reset()

    def __next__(self):
        line = self.readline()
        if line:
            return line
        raise StopIteration

    def __iter__(self):
        return self

    def __getattr__(self, name, getattr=getattr):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()


class StreamReaderWriter:
    encoding = "unknown"

    def __init__(self, stream, Reader, Writer, errors="strict"):
        self.stream = stream
        self.reader = Reader(stream, errors)
        self.writer = Writer(stream, errors)
        self.errors = errors

    def read(self, size=-1):
        return self.reader.read(size)

    def readline(self, size=None):
        return self.reader.readline(size)

    def readlines(self, sizehint=None):
        return self.reader.readlines(sizehint)

    def __next__(self):
        return next(self.reader)

    def __iter__(self):
        return self

    def write(self, data):
        return self.writer.write(data)

    def writelines(self, list):
        return self.writer.writelines(list)

    def reset(self):
        self.reader.reset()
        self.writer.reset()

    def seek(self, offset, whence=0):
        self.stream.seek(offset, whence)
        self.reader.reset()
        if whence == 0 and offset == 0:
            self.writer.reset()

    def __getattr__(self, name, getattr=getattr):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()


class StreamRecoder:
    data_encoding = "unknown"
    file_encoding = "unknown"

    def __init__(self, stream, encode, decode, Reader, Writer, errors="strict"):
        self.stream = stream
        self.encode = encode
        self.decode = decode
        self.reader = Reader(stream, errors)
        self.writer = Writer(stream, errors)
        self.errors = errors

    def read(self, size=-1):
        data = self.reader.read(size)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def readline(self, size=None):
        data = self.reader.readline() if size is None else self.reader.readline(size)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def readlines(self, sizehint=None):
        data = self.reader.read()
        data, bytesencoded = self.encode(data, self.errors)
        return data.splitlines(keepends=True)

    def __next__(self):
        data = next(self.reader)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def __iter__(self):
        return self

    def write(self, data):
        data, bytesdecoded = self.decode(data, self.errors)
        return self.writer.write(data)

    def writelines(self, list):
        data = b"".join(list)
        data, bytesdecoded = self.decode(data, self.errors)
        return self.writer.write(data)

    def reset(self):
        self.reader.reset()
        self.writer.reset()

    def seek(self, offset, whence=0):
        self.reader.seek(offset, whence)
        self.writer.seek(offset, whence)

    def __getattr__(self, name, getattr=getattr):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()


# ---- the builtin codecs ----


def _text_codec(name, python_name):
    def encode(input, errors="strict"):
        return str(input).encode(python_name, errors), len(input)

    def decode(input, errors="strict"):
        data = bytes(input)
        return data.decode(python_name, errors), len(data)

    class _Encoder(IncrementalEncoder):
        def encode(self, input, final=False):
            return str(input).encode(python_name, self.errors)

    class _Decoder(BufferedIncrementalDecoder):
        def _buffer_decode(self, input, errors, final):
            if python_name == "utf-8" and not final:
                # Keep an incomplete trailing multi-byte sequence for the next call.
                cut = len(input)
                tail = 0
                while tail < 3 and cut > 0 and input[cut - 1] & 0xC0 == 0x80:
                    cut -= 1
                    tail += 1
                if cut > 0 and input[cut - 1] >= 0xC0:
                    lead = input[cut - 1]
                    need = 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
                    if tail + 1 < need:
                        input = input[:cut - 1]
            return bytes(input).decode(python_name, errors), len(input)

    class _Reader(StreamReader):
        def decode(self, input, errors="strict"):
            return decode(input, errors)

    class _Writer(StreamWriter):
        def encode(self, input, errors="strict"):
            return encode(input, errors)

    return CodecInfo(encode, decode, _Reader, _Writer, _Encoder, _Decoder, name)


_ROT13 = {}
for _upper, _lower in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"):
    _ROT13[ord(_upper)] = ord("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[(ord(_upper) - 65 + 13) % 26])
    _ROT13[ord(_lower)] = ord("abcdefghijklmnopqrstuvwxyz"[(ord(_lower) - 97 + 13) % 26])


def _rot13_codec():
    def transform(input, errors="strict"):
        return str(input).translate(_ROT13), len(input)

    class _Encoder(IncrementalEncoder):
        def encode(self, input, final=False):
            return str(input).translate(_ROT13)

    class _Decoder(IncrementalDecoder):
        def decode(self, input, final=False):
            return str(input).translate(_ROT13)

    class _Reader(StreamReader):
        def decode(self, input, errors="strict"):
            return transform(input, errors)

    class _Writer(StreamWriter):
        def encode(self, input, errors="strict"):
            return transform(input, errors)

    return CodecInfo(transform, transform, _Reader, _Writer, _Encoder, _Decoder, "rot-13",
                     _is_text_encoding=False)


_ALIASES = {
    "utf_8": "utf-8", "utf8": "utf-8", "u8": "utf-8", "utf": "utf-8", "cp65001": "utf-8",
    "ascii": "ascii", "us_ascii": "ascii", "646": "ascii", "ansi_x3.4_1968": "ascii",
    "latin_1": "latin-1", "latin1": "latin-1", "iso8859_1": "latin-1", "iso_8859_1": "latin-1",
    "8859": "latin-1", "cp819": "latin-1", "l1": "latin-1",
    "rot_13": "rot-13", "rot13": "rot-13",
}
_BUILTIN = {
    "utf-8": lambda: _text_codec("utf-8", "utf-8"),
    "ascii": lambda: _text_codec("ascii", "ascii"),
    "latin-1": lambda: _text_codec("latin-1", "latin-1"),
    "rot-13": _rot13_codec,
}
_search_functions = []
_cache = {}


def _normalize(encoding):
    return encoding.lower().replace("-", "_").replace(" ", "_")


def register(search_function):
    if not callable(search_function):
        raise TypeError("argument must be callable")
    _search_functions.append(search_function)


def unregister(search_function):
    try:
        _search_functions.remove(search_function)
    except ValueError:
        pass
    _cache.clear()


def lookup(encoding):
    if not isinstance(encoding, str):
        raise TypeError("lookup() argument must be str, not " + type(encoding).__name__)
    key = _normalize(encoding)
    info = _cache.get(key)
    if info is not None:
        return info
    canonical = _ALIASES.get(key)
    if canonical is not None:
        info = _BUILTIN[canonical]()
    else:
        for search in _search_functions:
            info = search(key)
            if info is not None:
                if not isinstance(info, tuple) or len(info) != 4:
                    raise TypeError("codec search functions must return 4-tuples")
                break
        else:
            raise LookupError("unknown encoding: " + encoding)
    _cache[key] = info
    return info


def encode(obj, encoding="utf-8", errors="strict"):
    return lookup(encoding).encode(obj, errors)[0]


def decode(obj, encoding="utf-8", errors="strict"):
    return lookup(encoding).decode(obj, errors)[0]


def getencoder(encoding):
    return lookup(encoding).encode


def getdecoder(encoding):
    return lookup(encoding).decode


def getincrementalencoder(encoding):
    encoder = lookup(encoding).incrementalencoder
    if encoder is None:
        raise LookupError(encoding)
    return encoder


def getincrementaldecoder(encoding):
    decoder = lookup(encoding).incrementaldecoder
    if decoder is None:
        raise LookupError(encoding)
    return decoder


def getreader(encoding):
    return lookup(encoding).streamreader


def getwriter(encoding):
    return lookup(encoding).streamwriter


def iterencode(iterator, encoding, errors="strict", **kwargs):
    encoder = getincrementalencoder(encoding)(errors, **kwargs)
    for input in iterator:
        output = encoder.encode(input)
        if output:
            yield output
    output = encoder.encode("", True)
    if output:
        yield output


def iterdecode(iterator, encoding, errors="strict", **kwargs):
    decoder = getincrementaldecoder(encoding)(errors, **kwargs)
    for input in iterator:
        output = decoder.decode(input)
        if output:
            yield output
    output = decoder.decode(b"", True)
    if output:
        yield output


def open(filename, mode="r", encoding=None, errors="strict", buffering=-1):
    if encoding is not None and "b" not in mode:
        mode = mode + "b"
    file = _io.open(filename, mode)
    if encoding is None:
        return file
    try:
        info = lookup(encoding)
        srw = StreamReaderWriter(file, info.streamreader, info.streamwriter, errors)
        srw.encoding = encoding
        return srw
    except BaseException:
        file.close()
        raise


def EncodedFile(file, data_encoding, file_encoding=None, errors="strict"):
    if file_encoding is None:
        file_encoding = data_encoding
    data_info = lookup(data_encoding)
    file_info = lookup(file_encoding)
    sr = StreamRecoder(file, data_info.encode, data_info.decode,
                       file_info.streamreader, file_info.streamwriter, errors)
    sr.data_encoding = data_encoding
    sr.file_encoding = file_encoding
    return sr


# ---- error handlers ----

_error_handlers = {}


def register_error(name, handler):
    if not callable(handler):
        raise TypeError("handler must be callable")
    _error_handlers[name] = handler


def lookup_error(name):
    try:
        return _error_handlers[name]
    except KeyError:
        raise LookupError("unknown error handler name '%s'" % name) from None


def strict_errors(exc):
    raise exc


def _replacement_range(exc):
    return exc.object[exc.start:exc.end], exc.end


def ignore_errors(exc):
    if not isinstance(exc, UnicodeError):
        raise TypeError("don't know how to handle %s in error callback" % type(exc).__name__)
    return "", exc.end


def replace_errors(exc):
    if isinstance(exc, UnicodeEncodeError):
        return "?" * (exc.end - exc.start), exc.end
    if isinstance(exc, UnicodeDecodeError):
        return "�", exc.end
    if isinstance(exc, UnicodeTranslateError):
        return "�" * (exc.end - exc.start), exc.end
    raise TypeError("don't know how to handle %s in error callback" % type(exc).__name__)


def xmlcharrefreplace_errors(exc):
    if not isinstance(exc, UnicodeEncodeError):
        raise TypeError("don't know how to handle %s in error callback" % type(exc).__name__)
    text, end = _replacement_range(exc)
    return "".join("&#%d;" % ord(char) for char in text), end


def backslashreplace_errors(exc):
    if isinstance(exc, UnicodeDecodeError):
        return "".join("\\x%02x" % byte for byte in exc.object[exc.start:exc.end]), exc.end
    if not isinstance(exc, (UnicodeEncodeError, UnicodeTranslateError)):
        raise TypeError("don't know how to handle %s in error callback" % type(exc).__name__)
    text, end = _replacement_range(exc)
    pieces = []
    for char in text:
        code = ord(char)
        if code < 0x100:
            pieces.append("\\x%02x" % code)
        elif code < 0x10000:
            pieces.append("\\u%04x" % code)
        else:
            pieces.append("\\U%08x" % code)
    return "".join(pieces), end


def namereplace_errors(exc):
    if not isinstance(exc, UnicodeEncodeError):
        raise TypeError("don't know how to handle %s in error callback" % type(exc).__name__)
    return backslashreplace_errors(exc)


register_error("strict", strict_errors)
register_error("ignore", ignore_errors)
register_error("replace", replace_errors)
register_error("xmlcharrefreplace", xmlcharrefreplace_errors)
register_error("backslashreplace", backslashreplace_errors)
register_error("namereplace", namereplace_errors)
