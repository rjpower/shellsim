"""The `.npy` file format: NumPy's own NEP 1 layout, implemented from its published spec
(https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

A `.npy` file is a magic prefix, a version, a length-prefixed Python-dict-literal header
naming the dtype, shape, and memory order, padded so the header block ends on a 64-byte
boundary, followed by the raw element bytes (or, for `dtype=object`, a pickle of the whole
array; see `numpy.lib._objectpickle`). Structured dtypes are not part of shellsim's NumPy, so
a header whose `descr` is a list is rejected explicitly rather than partially supported.
"""

import numpy as np

from numpy.lib import _objectpickle
from numpy.lib._objectpickle import _pack_uint_le, _unpack_uint_le

MAGIC_PREFIX = b"\x93NUMPY"
ARRAY_ALIGN = 64


def magic(major, minor):
    if not (0 <= major < 256 and 0 <= minor < 256):
        raise ValueError("major and minor version must be in the range [0, 255]")
    return MAGIC_PREFIX + bytes([major, minor])


def read_magic(fp):
    data = fp.read(8)
    if len(data) != 8 or data[:6] != MAGIC_PREFIX:
        raise ValueError("the file is not a valid .npy file")
    return data[6], data[7]


def write_magic(fp, major=1, minor=0):
    fp.write(magic(major, minor))


def dtype_to_descr(dtype):
    return dtype.str


def descr_to_dtype(descr):
    if isinstance(descr, str):
        return np.dtype(descr)
    raise NotImplementedError("structured dtypes are not supported by shellsim's NumPy")


def header_data_from_array_1_0(array):
    fortran_order = bool(array.flags.f_contiguous) and not bool(array.flags.c_contiguous)
    return {
        "descr": dtype_to_descr(array.dtype),
        "fortran_order": fortran_order,
        "shape": tuple(array.shape),
    }


def _header_text(d):
    descr = repr(d["descr"])
    fortran_order = repr(bool(d["fortran_order"]))
    shape = repr(tuple(d["shape"]))
    return "{'descr': " + descr + ", 'fortran_order': " + fortran_order + ", 'shape': " + shape + ", }"


def _write_array_header(fp, d, version):
    text = _header_text(d)
    major, minor = version
    length_field = 2 if major == 1 else 4
    prefix_len = 6 + 2 + length_field
    total = prefix_len + len(text) + 1
    padded = -(-total // ARRAY_ALIGN) * ARRAY_ALIGN
    body = (text + " " * (padded - total) + "\n").encode("latin-1")
    if length_field == 2 and len(body) > 0xFFFF:
        _write_array_header(fp, d, (2, 0))
        return
    fp.write(magic(major, minor))
    fp.write(_pack_uint_le(len(body), length_field))
    fp.write(body)


def write_array_header_1_0(fp, d):
    _write_array_header(fp, d, (1, 0))


def write_array_header_2_0(fp, d):
    _write_array_header(fp, d, (2, 0))


class _StructuredDescr:
    """A placeholder for a `[...]` structured `descr`, rejected once it reaches a dtype."""


def _split_top_level(text):
    parts = []
    depth = 0
    quote = None
    current = []
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    tail = "".join(current)
    if tail.strip():
        parts.append(tail)
    return parts


def _parse_quoted(text):
    quote = text[0]
    if not text.endswith(quote):
        raise ValueError("invalid array header: " + text)
    return text[1:-1]


def _parse_tuple(text):
    inner = text[1:-1].strip()
    if not inner:
        return ()
    items = [item.strip() for item in inner.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def _parse_header_value(text):
    text = text.strip()
    if text[:1] in ("'", '"'):
        return _parse_quoted(text)
    if text.startswith("("):
        return _parse_tuple(text)
    if text.startswith("["):
        return _StructuredDescr()
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError("cannot parse array header value: " + text)


def _parse_header_dict(text):
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("invalid array header: " + text)
    result = {}
    for entry in _split_top_level(text[1:-1]):
        entry = entry.strip()
        if not entry:
            continue
        key_text, _, value_text = entry.partition(":")
        result[_parse_quoted(key_text.strip())] = _parse_header_value(value_text)
    for required in ("descr", "fortran_order", "shape"):
        if required not in result:
            raise ValueError("array header missing " + repr(required))
    return result


def _read_array_header(fp, version, max_header_size=10000):
    major, _minor = version
    length_field = 2 if major == 1 else 4
    raw_length = fp.read(length_field)
    if len(raw_length) != length_field:
        raise ValueError("EOF reading the .npy header length")
    length = _unpack_uint_le(raw_length)
    if length > max_header_size:
        raise ValueError("array header is too large")
    header = fp.read(length)
    if len(header) != length:
        raise ValueError("EOF reading the .npy header")
    return _parse_header_dict(header.decode("latin-1"))


def read_array_header_1_0(fp):
    return _read_array_header(fp, (1, 0))


def read_array_header_2_0(fp):
    return _read_array_header(fp, (2, 0))


def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    array = np.asanyarray(array)
    is_object = array.dtype == np.dtype(object)
    if is_object and not allow_pickle:
        raise ValueError("Object arrays cannot be saved when allow_pickle=False")
    header = header_data_from_array_1_0(array)
    _write_array_header(fp, header, version or (1, 0))
    if is_object:
        fp.write(_objectpickle.dumps(array))
        return
    order = "F" if header["fortran_order"] else "C"
    fp.write(array.tobytes(order=order))


def read_array(fp, allow_pickle=False, pickle_kwargs=None, max_header_size=10000):
    major, minor = read_magic(fp)
    if major not in (1, 2, 3):
        raise ValueError(f"unsupported .npy version {major}.{minor}")
    header = _read_array_header(fp, (major, minor), max_header_size=max_header_size)
    dtype = descr_to_dtype(header["descr"])
    shape = header["shape"]
    fortran_order = header["fortran_order"]
    if dtype == np.dtype(object):
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
        return _objectpickle.loads(fp.read())
    count = 1
    for dim in shape:
        count *= dim
    nbytes = count * dtype.itemsize
    data = fp.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"EOF: reading array data, expected {nbytes} bytes, got {len(data)}")
    flat = np.frombuffer(data, dtype=dtype, count=count).copy()
    if not shape:
        return flat.reshape(())
    return flat.reshape(shape, order="F" if fortran_order else "C")
