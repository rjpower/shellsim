"""File I/O through the virtual filesystem: the `.npy`/`.npz` binary format and text readers and
writers.

The `.npy` layout follows NumPy's own published NEP 1 spec
(https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html): a magic prefix, a
version, a length-prefixed Python-dict-literal header naming the dtype, shape, and memory order,
padded so the header block ends on a 64-byte boundary, followed by the raw element bytes.
Structured dtypes are not part of shellsim's NumPy, so a header whose `descr` is a list is
rejected explicitly rather than partially supported. `dtype=object` arrays are rejected too:
shellsim has no `pickle` module and no clean-room pickler for them, so `save`/`load` raise
`ValueError` for that dtype the same way real NumPy does for `allow_pickle=False`, unconditionally.

`.npz` archives are plain ZIP files (via the frozen `zipfile` module) holding one `<name>.npy`
member per array. The text functions (`savetxt`, `loadtxt`, `genfromtxt`) are line-oriented so
their cost stays proportional to the text they read or write, per AGENTS.md's metering rule; they
build on ordinary Python string and float conversion rather than a bespoke numeric parser.
"""

import io
import zipfile

import numpy as np

MAGIC_PREFIX = b"\x93NUMPY"
ARRAY_ALIGN = 64


# -- the .npy format ------------------------------------------------------------------------


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
    fp.write(len(body).to_bytes(length_field, "little"))
    fp.write(body)


def write_array_header_1_0(fp, d):
    _write_array_header(fp, d, (1, 0))


def write_array_header_2_0(fp, d):
    _write_array_header(fp, d, (2, 0))


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
        # A structured `descr` is a `[...]` list; `descr_to_dtype` rejects it once it gets
        # there, so it only needs to survive parsing as an opaque marker.
        return []
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
    length = int.from_bytes(raw_length, "little")
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
    if array.dtype == np.dtype(object):
        raise ValueError("dtype=object arrays cannot be saved by shellsim's NumPy")
    header = header_data_from_array_1_0(array)
    _write_array_header(fp, header, version or (1, 0))
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
        raise ValueError("dtype=object arrays cannot be loaded by shellsim's NumPy")
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


# -- save/load/savez ------------------------------------------------------------------------


def _open_binary(file, mode):
    if hasattr(file, "read" if "r" in mode else "write"):
        return file, False
    return open(str(file), mode), True


def _with_suffix(path, suffix):
    path = str(path)
    return path if path.endswith(suffix) else path + suffix


def save(file, arr, allow_pickle=True, fix_imports=True):
    """Write one array to `file` (a path or a writable binary stream) as `.npy`."""
    array = np.asanyarray(arr)
    if hasattr(file, "write"):
        fp, own = file, False
    else:
        fp, own = open(_with_suffix(file, ".npy"), "wb"), True
    try:
        write_array(fp, array, allow_pickle=allow_pickle)
    finally:
        if own:
            fp.close()


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding="ASCII"):
    """Read one `.npy` array, or open an `.npz` archive as an `NpzFile`."""
    if mmap_mode is not None:
        raise NotImplementedError("memory-mapped arrays are not supported by shellsim's NumPy")
    fp, own = _open_binary(file, "rb")
    prefix = fp.read(2)
    fp.seek(-len(prefix), 1)
    if prefix == b"PK":
        return NpzFile(fp, own, allow_pickle)
    try:
        return read_array(fp, allow_pickle=allow_pickle)
    finally:
        if own:
            fp.close()


class NpzFile:
    """The lazily-loaded mapping `np.load` returns for an `.npz` archive."""

    def __init__(self, fp, own_file, allow_pickle):
        self._zip = zipfile.ZipFile(fp)
        self._fp = fp
        self._own_file = own_file
        self._allow_pickle = allow_pickle
        self._cache = {}
        self.files = [
            name[:-4] if name.endswith(".npy") else name for name in self._zip.namelist()
        ]

    def __contains__(self, key):
        return key in self.files

    def __iter__(self):
        return iter(self.files)

    def keys(self):
        return list(self.files)

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        if key not in self._cache:
            member = key + ".npy" if key + ".npy" in self._zip.namelist() else key
            data = self._zip.read(member)
            self._cache[key] = read_array(io.BytesIO(data), allow_pickle=self._allow_pickle)
        return self._cache[key]

    def close(self):
        self._zip.close()
        if self._own_file:
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False


def _savez(file, args, kwargs, compress):
    named = dict(kwargs)
    for index, array in enumerate(args):
        named[f"arr_{index}"] = array
    if hasattr(file, "write"):
        fp, own = file, False
    else:
        fp, own = open(_with_suffix(file, ".npz"), "wb"), True
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    try:
        with zipfile.ZipFile(fp, "w", compression=compression) as archive:
            for name, value in named.items():
                buffer = io.BytesIO()
                write_array(buffer, np.asanyarray(value))
                archive.writestr(name + ".npy", buffer.getvalue())
    finally:
        if own:
            fp.close()


def savez(file, *args, **kwargs):
    _savez(file, args, kwargs, compress=False)


def savez_compressed(file, *args, **kwargs):
    _savez(file, args, kwargs, compress=True)


# -- text I/O ---------------------------------------------------------------


def _open_text_source(fname):
    if hasattr(fname, "read"):
        return fname, False
    return open(str(fname), "r"), True


def _open_text_sink(fname):
    if hasattr(fname, "write"):
        return fname, False
    return open(str(fname), "w"), True


def _strip_comment(line, comments):
    if comments:
        index = line.find(comments)
        if index >= 0:
            line = line[:index]
    return line


def _split_fields(line, delimiter):
    return line.split() if delimiter is None else line.split(delimiter)


def _select_usecols(fields, usecols):
    indices = [usecols] if isinstance(usecols, int) else list(usecols)
    return [fields[index] for index in indices]


def _rows_to_array(rows, dtype):
    if not rows:
        return np.array([], dtype=dtype)
    ncols = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(
                f"the number of columns changed from {ncols} to {len(row)} at row {index}"
            )
    if len(rows) == 1:
        return np.array(rows[0], dtype=dtype)
    if ncols == 1:
        return np.array([row[0] for row in rows], dtype=dtype)
    return np.array(rows, dtype=dtype)


def loadtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    converters=None,
    skiprows=0,
    usecols=None,
    unpack=False,
    ndmin=0,
    encoding="bytes",
    max_rows=None,
    quotechar=None,
    like=None,
):
    dtype = np.dtype(dtype) if dtype is not None else np.float64
    fp, own = _open_text_source(fname)
    try:
        rows = []
        for index, raw_line in enumerate(fp):
            if index < skiprows:
                continue
            line = _strip_comment(raw_line.rstrip("\n"), comments)
            if not line.strip():
                continue
            fields = _split_fields(line, delimiter)
            if usecols is not None:
                fields = _select_usecols(fields, usecols)
            values = []
            for field in fields:
                try:
                    values.append(float(field))
                except ValueError:
                    raise ValueError(f"could not convert string {field!r} to float64")
            rows.append(values)
            if max_rows is not None and len(rows) >= max_rows:
                break
        return _rows_to_array(rows, dtype)
    finally:
        if own:
            fp.close()


def _format_conversion(fmt):
    for char in reversed(fmt):
        if char.isalpha():
            return char
    return "s"


def _coerce_for_fmt(fmt, value):
    conversion = _format_conversion(fmt)
    if conversion in "diouxX":
        return int(value)
    if conversion in "eEfFgG":
        return float(value)
    return value


def _row_formatters(fmt, ncols):
    if isinstance(fmt, (list, tuple)):
        if len(fmt) != ncols:
            raise ValueError(f"fmt has wrong shape: expected {ncols} formats, got {len(fmt)}")
        return list(fmt)
    if fmt.count("%") > 1:
        return [fmt]
    return [fmt] * ncols


def savetxt(
    fname,
    X,
    fmt="%.18e",
    delimiter=" ",
    newline="\n",
    header="",
    footer="",
    comments="# ",
    encoding=None,
):
    array = np.asarray(X)
    if array.ndim == 0:
        array = array.reshape(1)
    rows = array.tolist() if array.ndim > 1 else [[value] for value in array.tolist()]
    ncols = len(rows[0]) if rows else array.shape[-1] if array.ndim > 1 else 1
    formatters = _row_formatters(fmt, ncols)
    fp, own = _open_text_sink(fname)
    try:
        if header:
            fp.write(comments + header + newline)
        for row in rows:
            if len(formatters) == 1 and ncols > 1:
                line = formatters[0] % tuple(_coerce_for_fmt(formatters[0], v) for v in row)
            else:
                line = delimiter.join(
                    f % _coerce_for_fmt(f, v) for f, v in zip(formatters, row)
                )
            fp.write(line + newline)
        if footer:
            fp.write(comments + footer + newline)
    finally:
        if own:
            fp.close()


def _column_is_numeric(rows, column):
    for row in rows:
        value = row[column].strip()
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
    return True


def _parse_missing(value):
    value = value.strip()
    return float("nan") if value == "" else float(value)


def genfromtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    skip_header=0,
    skip_footer=0,
    usecols=None,
    names=None,
    usemask=False,
    missing_values=None,
    filling_values=None,
    encoding="bytes",
    **kwargs,
):
    if names:
        raise NotImplementedError("genfromtxt with names= builds structured arrays")
    if usemask:
        raise NotImplementedError("masked arrays are not supported by shellsim's NumPy")
    fp, own = _open_text_source(fname)
    try:
        rows = []
        for index, raw_line in enumerate(fp):
            if index < skip_header:
                continue
            line = _strip_comment(raw_line.rstrip("\n"), comments)
            if not line.strip():
                continue
            fields = _split_fields(line, delimiter)
            if usecols is not None:
                fields = _select_usecols(fields, usecols)
            rows.append(fields)
        if skip_footer:
            rows = rows[:-skip_footer]
        if not rows:
            return np.array([], dtype=(dtype if dtype is not None else np.float64))
        ncols = len(rows[0])
        for index, row in enumerate(rows):
            if len(row) != ncols:
                raise ValueError(
                    f"the number of columns changed from {ncols} to {len(row)} at row {index}"
                )
        if dtype is None:
            numeric_flags = {_column_is_numeric(rows, column) for column in range(ncols)}
            if len(numeric_flags) > 1:
                raise NotImplementedError(
                    "genfromtxt columns of different types need a structured array"
                )
            resolved_dtype = np.float64
        else:
            resolved_dtype = np.dtype(dtype)
        parsed = [[_parse_missing(value) for value in row] for row in rows]
        return _rows_to_array(parsed, resolved_dtype)
    finally:
        if own:
            fp.close()
