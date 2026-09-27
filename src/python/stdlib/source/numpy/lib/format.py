"""``numpy.lib.format``: the ``.npy`` file format, versions 1.0 to 3.0.

A file is the magic string ``\\x93NUMPY``, a two-byte version, a little-endian header length,
and a header holding a Python dict literal with ``descr``, ``fortran_order`` and ``shape``,
padded with spaces so the data starts on a 64-byte boundary and ended by a newline. The array's
bytes follow in C or Fortran order, in the byte order ``descr`` names.

shellsim has no ``ast``, so headers are read by a small parser for the literals a header can
contain. Object arrays are pickled by ``numpy.lib._objectpickle``, which loads only NumPy array
pickles. Structured dtypes and memory maps are not supported.
"""

import struct
import warnings

import numpy
from numpy.lib import _objectpickle

EXPECTED_KEYS = {"descr", "fortran_order", "shape"}
MAGIC_PREFIX = b"\x93NUMPY"
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ARRAY_ALIGN = 64
BUFFER_SIZE = 2**18
GROWTH_AXIS_MAX_DIGITS = 21

_header_size_info = {
    (1, 0): ("<H", "latin1"),
    (2, 0): ("<I", "latin1"),
    (3, 0): ("<I", "utf8"),
}
_MAX_HEADER_SIZE = 10000


def _check_version(version):
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))


def magic(major, minor):
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([major, minor])


def read_magic(fp):
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    return magic_str[-2], magic_str[-1]


def dtype_to_descr(dtype):
    return dtype.str


def descr_to_dtype(descr):
    if isinstance(descr, str):
        return numpy.dtype(descr)
    raise NotImplementedError("structured dtypes are not supported by shellsim's NumPy")


def header_data_from_array_1_0(array):
    d = {"shape": array.shape}
    if array.flags.c_contiguous:
        d["fortran_order"] = False
    elif array.flags.f_contiguous:
        d["fortran_order"] = True
    else:
        # A non-contiguous array is written in C order.
        d["fortran_order"] = False
    d["descr"] = dtype_to_descr(array.dtype)
    return d


def _wrap_header(header, version):
    fmt, encoding = _header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)
    if hlen + padlen >= 1 << (8 * struct.calcsize(fmt)):
        raise ValueError(f"Header length {hlen} too big for version={version}")
    header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    return header_prefix + header + b" " * padlen + b"\n"


def _wrap_header_guess_version(header):
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass
    try:
        wrapped = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn(
            "Stored array in format 2.0. It can only be read by NumPy >= 1.9",
            UserWarning,
            stacklevel=2,
        )
        return wrapped
    wrapped = _wrap_header(header, (3, 0))
    warnings.warn(
        "Stored array in format 3.0. It can only be read by NumPy >= 1.17",
        UserWarning,
        stacklevel=2,
    )
    return wrapped


def _write_array_header(fp, d, version=None):
    header = ["{"]
    for key, value in sorted(d.items()):
        header.append(f"'{key}': {repr(value)}, ")
    header.append("}")
    header = "".join(header)
    # Spare room lets the header be rewritten in place when the array grows along its
    # slowest axis.
    shape = d["shape"]
    if len(shape) > 0:
        growth = shape[-1 if d["fortran_order"] else 0]
        header += " " * (GROWTH_AXIS_MAX_DIGITS - len(repr(growth)))
    if version is None:
        header = _wrap_header_guess_version(header)
    else:
        header = _wrap_header(header, version)
    fp.write(header)


def write_array_header_1_0(fp, d):
    _write_array_header(fp, d, (1, 0))


def write_array_header_2_0(fp, d):
    _write_array_header(fp, d, (2, 0))


def read_array_header_1_0(fp, max_header_size=_MAX_HEADER_SIZE):
    return _read_array_header(fp, version=(1, 0), max_header_size=max_header_size)


def read_array_header_2_0(fp, max_header_size=_MAX_HEADER_SIZE):
    return _read_array_header(fp, version=(2, 0), max_header_size=max_header_size)


class _LiteralReader:
    """Parse the Python literals a header can hold: dicts, tuples, lists, strings, ints (with
    Python 2's optional ``L`` suffix), and ``True``, ``False`` and ``None``."""

    _ESCAPES = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "t": "\t", "r": "\r", "0": "\0"}

    def __init__(self, text):
        self.text = text
        self.index = 0

    def fail(self):
        raise ValueError(f"Cannot parse header: {self.text!r}")

    def skip_space(self):
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def peek(self):
        self.skip_space()
        if self.index >= len(self.text):
            self.fail()
        return self.text[self.index]

    def expect(self, char):
        if self.peek() != char:
            self.fail()
        self.index += 1

    def document(self):
        value = self.value()
        self.skip_space()
        if self.index != len(self.text):
            self.fail()
        return value

    def value(self):
        char = self.peek()
        if char == "{":
            return self.mapping()
        if char in "([":
            return self.sequence(char)
        if char in "'\"":
            return self.string(char)
        if char.isdigit() or char in "+-":
            return self.integer()
        for word, value in (("True", True), ("False", False), ("None", None)):
            if self.text.startswith(word, self.index):
                self.index += len(word)
                return value
        self.fail()

    def mapping(self):
        self.expect("{")
        result = {}
        while self.peek() != "}":
            key = self.value()
            self.expect(":")
            result[key] = self.value()
            if self.peek() == ",":
                self.index += 1
            elif self.peek() != "}":
                self.fail()
        self.index += 1
        return result

    def sequence(self, opening):
        closing = ")" if opening == "(" else "]"
        self.expect(opening)
        items = []
        trailing_comma = False
        while self.peek() != closing:
            items.append(self.value())
            trailing_comma = False
            if self.peek() == ",":
                self.index += 1
                trailing_comma = True
            elif self.peek() != closing:
                self.fail()
        self.index += 1
        if opening == "[":
            return items
        if len(items) == 1 and not trailing_comma:
            return items[0]
        return tuple(items)

    def string(self, quote):
        self.index += 1
        chars = []
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == quote:
                return "".join(chars)
            if char == "\\":
                if self.index >= len(self.text):
                    self.fail()
                escaped = self._ESCAPES.get(self.text[self.index])
                if escaped is None:
                    self.fail()
                chars.append(escaped)
                self.index += 1
            else:
                chars.append(char)
        self.fail()

    def integer(self):
        start = self.index
        if self.text[self.index] in "+-":
            self.index += 1
        digits = self.index
        while self.index < len(self.text) and self.text[self.index].isdigit():
            self.index += 1
        if self.index == digits:
            self.fail()
        value = int(self.text[start:self.index])
        if self.index < len(self.text) and self.text[self.index] == "L":
            self.index += 1
        return value


def _read_array_header(fp, version, max_header_size=_MAX_HEADER_SIZE):
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError(f"Invalid version {version!r}")
    hlength_type, encoding = hinfo
    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_bytes(fp, header_length, "array header")
    header = header.decode(encoding)
    if len(header) > max_header_size:
        raise ValueError(
            f"Header info length ({len(header)}) is large and may not be safe "
            "to load securely.\n"
            "To allow loading, adjust `max_header_size` or fully trust "
            "the `.npy` file using `allow_pickle=True`.\n"
            "For safety against large resource use or crashes, sandboxing "
            "may be necessary."
        )
    d = _LiteralReader(header).document()
    if not isinstance(d, dict):
        raise ValueError(f"Header is not a dictionary: {d!r}")
    if EXPECTED_KEYS != set(d.keys()):
        keys = sorted(d.keys())
        raise ValueError(f"Header does not contain the correct keys: {keys!r}")
    shape = d["shape"]
    if not isinstance(shape, tuple) or not all(type(x) is int for x in shape):
        raise ValueError(f"shape is not valid: {shape!r}")
    if not isinstance(d["fortran_order"], bool):
        raise ValueError(f"fortran_order is not a valid bool: {d['fortran_order']!r}")
    try:
        dtype = descr_to_dtype(d["descr"])
    except TypeError as e:
        raise ValueError(f"descr is not a valid dtype descriptor: {d['descr']!r}") from e
    return shape, d["fortran_order"], dtype


def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)
    if array.dtype.hasobject:
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when allow_pickle=False")
        _objectpickle.dump(array, fp)
        return
    if array.itemsize == 0:
        return
    order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    # Write in 16 MiB pieces, as NumPy's buffered iterator does, so the copy made for
    # ``tobytes`` stays bounded.
    flat = array.ravel(order=order)
    step = max(16 * 1024**2 // array.itemsize, 1)
    for start in range(0, flat.size, step):
        fp.write(flat[start:start + step].tobytes())


def read_array(fp, allow_pickle=False, pickle_kwargs=None, *, max_header_size=_MAX_HEADER_SIZE):
    if allow_pickle:
        # A trusted file may have any header size.
        max_header_size = 2**64
    version = read_magic(fp)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(fp, version, max_header_size=max_header_size)
    count = 1
    for length in shape:
        count *= length
    if dtype.hasobject:
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
        encoding = (pickle_kwargs or {}).get("encoding", "ASCII")
        return _objectpickle.load(fp, encoding)
    array = numpy.empty(count, dtype=dtype)
    # Read 256 KiB at a time so a large file never needs a second full-size buffer.
    max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, max(dtype.itemsize, 1))
    for i in range(0, count if dtype.itemsize else 0, max_read_count):
        read_count = min(max_read_count, count - i)
        data = _read_bytes(fp, read_count * dtype.itemsize, "array data")
        array[i:i + read_count] = numpy.frombuffer(data, dtype=dtype, count=read_count)
    if fortran_order:
        return array.reshape(shape[::-1]).transpose()
    return array.reshape(shape)


def _read_bytes(fp, size, error_template="ran out of data"):
    data = b""
    while True:
        r = fp.read(size - len(data))
        data += r
        if len(r) == 0 or len(data) == size:
            break
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    return data


def isfileobj(f):
    return False


def open_memmap(filename, mode="r+", dtype=None, shape=None, fortran_order=False,
                version=None, *, max_header_size=_MAX_HEADER_SIZE):
    raise NotImplementedError("memory-mapped arrays are not supported by shellsim's NumPy")
