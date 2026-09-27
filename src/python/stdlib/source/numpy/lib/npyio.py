"""File I/O through the virtual filesystem: `.npy`/`.npz` binary I/O and text readers/writers.

Binary array persistence (`save`, `load`, `savez`, `savez_compressed`) is a thin layer over
`numpy.lib.format`: `.npz` archives are plain ZIP files (via the frozen `zipfile` module)
holding one `<name>.npy` member per array. The text functions (`savetxt`, `loadtxt`,
`genfromtxt`) are line-oriented so their cost stays proportional to the text they read or
write, per AGENTS.md's metering rule; they build on ordinary Python string and float
conversion rather than a bespoke numeric parser.
"""

import io
import zipfile

import numpy as np

from numpy.lib import format


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
        format.write_array(fp, array, allow_pickle=allow_pickle)
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
        return format.read_array(fp, allow_pickle=allow_pickle)
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
            self._cache[key] = format.read_array(io.BytesIO(data), allow_pickle=self._allow_pickle)
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
                format.write_array(buffer, np.asanyarray(value), allow_pickle=True)
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
