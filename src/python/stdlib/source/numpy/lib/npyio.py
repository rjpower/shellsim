"""``numpy.lib.npyio``: ``save``, ``load``, ``.npz`` archives, and text files.

Files go through the simulated ``open``, and ``.npz`` archives through the frozen ``zipfile``,
so every read and write stays inside the virtual filesystem. ``loadtxt`` and ``savetxt`` follow
NumPy's parser rules and error messages. ``genfromtxt`` supports results of one dtype, given or
inferred; field names, masks, and mixed inferred column types need structured arrays and raise
``NotImplementedError``.
"""

import operator
import os
import warnings

import numpy as np
from numpy.lib import _objectpickle, format

__all__ = [
    "genfromtxt",
    "load",
    "loadtxt",
    "NpzFile",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
]


class ConversionWarning(UserWarning):
    pass


class BagObj:
    """``NpzFile.f``: attribute access to the arrays of an archive."""

    def __init__(self, npz):
        self._npz = npz

    def __getattr__(self, key):
        try:
            return self._npz[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        return list(self._npz.keys())


def zipfile_factory(file, *args, **kwargs):
    if not hasattr(file, "read"):
        file = os.fspath(file)
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, *args, **kwargs)


class NpzFile:
    """The arrays of an ``.npz`` archive, loaded on access by name.

    ``files`` lists the names without their ``.npy`` suffix. Use it as a context manager, or
    call ``close``, to release the archive.
    """

    zip = None
    fid = None
    _MAX_REPR_ARRAY_COUNT = 5

    def __init__(self, fid, own_fid=False, allow_pickle=False, pickle_kwargs=None, *,
                 max_header_size=format._MAX_HEADER_SIZE):
        _zip = zipfile_factory(fid)
        _files = _zip.namelist()
        self.files = [name.removesuffix(".npy") for name in _files]
        self._files = dict(zip(self.files, _files))
        self._files.update(zip(_files, _files))
        self.allow_pickle = allow_pickle
        self.max_header_size = max_header_size
        self.pickle_kwargs = pickle_kwargs
        self.zip = _zip
        self.f = BagObj(self)
        if own_fid:
            self.fid = fid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        try:
            key = self._files[key]
        except KeyError:
            raise KeyError(f"{key} is not a file in the archive") from None
        with self.zip.open(key) as member:
            magic = member.read(len(format.MAGIC_PREFIX))
            member.seek(0)
            if magic == format.MAGIC_PREFIX:
                return format.read_array(
                    member,
                    allow_pickle=self.allow_pickle,
                    pickle_kwargs=self.pickle_kwargs,
                    max_header_size=self.max_header_size,
                )
            return member.read()

    def __contains__(self, key):
        return key in self._files

    def __repr__(self):
        if isinstance(self.fid, str):
            filename = self.fid
        else:
            filename = getattr(self.fid, "name", "object")
        array_names = ", ".join(self.files[:self._MAX_REPR_ARRAY_COUNT])
        if len(self.files) > self._MAX_REPR_ARRAY_COUNT:
            array_names += "..."
        return f"NpzFile {filename!r} with keys: {array_names}"

    def get(self, key, default=None, /):
        return self[key] if key in self else default

    def keys(self):
        return list(self.files)

    def values(self):
        return [self[key] for key in self.files]

    def items(self):
        return [(key, self[key]) for key in self.files]


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding="ASCII", *,
         max_header_size=format._MAX_HEADER_SIZE):
    if encoding not in ("ASCII", "latin1", "bytes"):
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")
    if mmap_mode:
        raise NotImplementedError("memory-mapped arrays are not supported by shellsim's NumPy")
    pickle_kwargs = {"encoding": encoding, "fix_imports": fix_imports}
    if hasattr(file, "read"):
        fid = file
        own_fid = False
    else:
        fid = open(os.fspath(file), "rb")
        own_fid = True
    try:
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        if not magic:
            raise EOFError("No data left in file")
        fid.seek(-min(N, len(magic)), 1)
        if magic.startswith((b"PK\x03\x04", b"PK\x05\x06")):
            archive = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle,
                              pickle_kwargs=pickle_kwargs, max_header_size=max_header_size)
            own_fid = False
            return archive
        if magic == format.MAGIC_PREFIX:
            return format.read_array(fid, allow_pickle=allow_pickle,
                                     pickle_kwargs=pickle_kwargs,
                                     max_header_size=max_header_size)
        if not allow_pickle:
            raise ValueError(
                "This file contains pickled (object) data. If you trust "
                "the file you can load it unsafely using the "
                "`allow_pickle=` keyword argument or `pickle.load()`."
            )
        try:
            return _objectpickle.load(fid, encoding)
        except Exception as e:
            raise _objectpickle.UnpicklingError(
                f"Failed to interpret file {file!r} as a pickle"
            ) from e
    finally:
        if own_fid:
            fid.close()


def save(file, arr, allow_pickle=True):
    if hasattr(file, "write"):
        fid = file
        own_fid = False
    else:
        file = os.fspath(file)
        if not file.endswith(".npy"):
            file = file + ".npy"
        fid = open(file, "wb")
        own_fid = True
    try:
        format.write_array(fid, np.asanyarray(arr), allow_pickle=allow_pickle)
    finally:
        if own_fid:
            fid.close()


def savez(file, *args, allow_pickle=True, **kwds):
    _savez(file, args, kwds, False, allow_pickle=allow_pickle)


def savez_compressed(file, *args, allow_pickle=True, **kwds):
    _savez(file, args, kwds, True, allow_pickle=allow_pickle)


def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    import zipfile

    if not hasattr(file, "write"):
        file = os.fspath(file)
        if not file.endswith(".npz"):
            file = file + ".npz"
    namedict = kwds
    for i, val in enumerate(args):
        key = f"arr_{i}"
        if key in namedict.keys():
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
        namedict[key] = val
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    zipf = zipfile_factory(file, mode="w", compression=compression)
    try:
        for key, val in namedict.items():
            with zipf.open(key + ".npy", "w", force_zip64=True) as fid:
                format.write_array(fid, np.asanyarray(val), allow_pickle=allow_pickle,
                                   pickle_kwargs=pickle_kwargs)
    finally:
        zipf.close()


def _check_nonneg_int(value, name="argument"):
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _ensure_ndmin_ndarray_check_param(ndmin):
    if ndmin not in [0, 1, 2]:
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")


def _ensure_ndmin_ndarray(a, *, ndmin):
    if a.ndim > ndmin:
        a = np.squeeze(a)
    if a.ndim < ndmin:
        if ndmin == 1:
            a = np.atleast_1d(a)
        elif ndmin == 2:
            a = np.atleast_2d(a).T
    return a


def _decode_line(line, encoding=None):
    if type(line) is bytes:
        return line.decode(encoding or "latin1")
    return line


def _split_text(text):
    """Split file text into lines as universal-newline reading does, without line endings."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _source_lines(fname, encoding):
    """The lines of a path, a file object, or an iterable of lines, without line endings."""
    if hasattr(fname, "__fspath__"):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        with open(fname, "rt", encoding=encoding) as handle:
            return _split_text(handle.read())
    if hasattr(fname, "read"):
        return _split_text(_decode_line(fname.read(), encoding))
    try:
        items = iter(fname)
    except TypeError as e:
        raise ValueError(
            "fname must be a string, filehandle, list of strings,\n"
            f"or generator. Got {type(fname)} instead."
        ) from e
    lines = []
    for item in items:
        lines.extend(_split_text(_decode_line(item, encoding)) or [""])
    return lines


def _check_control_character(value):
    if value is not None and (not isinstance(value, str) or len(value) != 1):
        raise TypeError(
            "Text reading control character must be a single unicode character or None; "
            f"but got: {value!r}"
        )


def _fields(line, delimiter, comments, quote):
    """Split one line into fields after removing its comment; ``[]`` means no data."""
    if quote is None:
        cut = len(line)
        for comment in comments:
            index = line.find(comment)
            if 0 <= index < cut:
                cut = index
        line = line[:cut]
        if delimiter is None:
            return line.split()
        return line.split(delimiter) if line else []
    fields = []
    field = []
    started = False
    quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if quoted:
            if char == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    field.append(quote)
                    index += 1
                else:
                    quoted = False
            else:
                field.append(char)
        elif char == quote:
            quoted = True
            started = True
        elif comments and line.startswith(comments[0], index):
            break
        elif delimiter is None and char in " \t":
            if started:
                fields.append("".join(field))
                field = []
                started = False
        elif char == delimiter:
            fields.append("".join(field))
            field = []
            started = False
        else:
            field.append(char)
            started = True
        index += 1
    if started or field or (delimiter is not None and fields):
        fields.append("".join(field))
    return fields


def _number_parser(dtype):
    """Parse one field as NumPy's text reader does for ``dtype``; raise ``ValueError`` if it
    does not hold a value of that dtype."""
    kind = dtype.kind
    if kind in "iub":
        limits = (0, 1) if kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)

        def parse(text):
            text = text.strip()
            digits = text[1:] if text[:1] in "+-" else text
            if not digits.isdigit() or not digits.isascii():
                raise ValueError(text)
            value = int(text)
            if kind == "b":
                return value != 0
            if not limits[0] <= value <= limits[1]:
                raise ValueError(text)
            return value

        return parse
    if kind in "fc":
        convert = float if kind == "f" else complex

        def parse(text):
            if "_" in text:
                raise ValueError(text)
            return convert(text.strip())

        return parse
    return str


def _read(fname, *, delimiter=",", comment="#", quote='"', usecols=None, skiplines=0,
          max_rows=None, converters=None, ndmin=None, unpack=False, dtype=np.float64,
          encoding=None):
    if encoding == "bytes":
        encoding = None
    if dtype is None:
        raise TypeError("a dtype must be provided.")
    dtype = np.dtype(dtype)
    if usecols is not None:
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]
        for column in usecols:
            operator.index(column)
    _ensure_ndmin_ndarray_check_param(ndmin)
    if comment is None:
        comments = []
    else:
        if "" in comment:
            raise ValueError(
                "comments cannot be an empty string. Use comments=None to disable comments."
            )
        comments = list(comment)
        if len(comments) > 1 and delimiter in comments:
            raise TypeError(
                f"Comment characters '{comments}' cannot include the delimiter '{delimiter}'"
            )
        if quote is not None and (len(comments) > 1 or len(comments[0]) > 1):
            raise ValueError(
                "when multiple comments or a multi-character comment is "
                "given, quotes are not supported.  In this case quotechar "
                "must be set to None."
            )
    _check_control_character(delimiter)
    _check_control_character(quote)
    _check_nonneg_int(skiplines)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    lines = _source_lines(fname, encoding)
    parse = _number_parser(dtype)
    column_converters = {}
    rows = []
    ncols = None
    for line in lines[skiplines:]:
        if max_rows is not None and len(rows) >= max_rows:
            break
        fields = _fields(line, delimiter, comments, quote)
        if not fields:
            continue
        row = len(rows)
        if usecols is None:
            if ncols is None:
                ncols = len(fields)
            elif len(fields) != ncols:
                raise ValueError(
                    f"the number of columns changed from {ncols} to {len(fields)} at row "
                    f"{row + 1}; use `usecols` to select a subset and avoid this error"
                )
            columns = range(len(fields))
        else:
            columns = []
            for column in usecols:
                index = column + len(fields) if column < 0 else column
                if not 0 <= index < len(fields):
                    raise ValueError(
                        f"invalid column index {column} at row {row + 1} "
                        f"with {len(fields)} columns"
                    )
                columns.append(index)
        values = []
        for column in columns:
            text = fields[column]
            converter = column_converters.get(column)
            if converter is None:
                converter = _column_converter(converters, column, len(fields), usecols)
                column_converters[column] = converter
            if converter is not None:
                values.append(converter(text))
                continue
            try:
                values.append(parse(text))
            except ValueError:
                raise ValueError(
                    f"could not convert string {text!r} to {dtype.name} at row {row}, "
                    f"column {column + 1}."
                ) from None
        rows.append(values)
    width = len(usecols) if usecols is not None else (ncols or 1)
    arr = np.array([value for values in rows for value in values], dtype=dtype)
    arr = arr.reshape(len(rows), width)
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)
    if arr.shape and arr.shape[0] == 0:
        warnings.warn(f'loadtxt: input contained no data: "{fname}"', category=UserWarning,
                      stacklevel=3)
    if unpack:
        return arr.T
    return arr


def _column_converter(converters, column, nfields, usecols):
    """The user converter for a file column: one callable for all columns, or a dict keyed by
    column number, negative numbers counting from the end."""
    if converters is None:
        return None
    if callable(converters):
        return converters
    for key, converter in converters.items():
        index = key + nfields if key < 0 else key
        if index == column:
            return converter
    return None


def loadtxt(fname, dtype=float, comments="#", delimiter=None, converters=None, skiprows=0,
            usecols=None, unpack=False, ndmin=0, encoding=None, max_rows=None, *,
            quotechar=None, like=None):
    if dtype is None:
        dtype = np.float64
    comment = comments
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [x.decode("latin1") if isinstance(x, bytes) else x for x in comment]
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode("latin1")
    return _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
                 converters=converters, skiplines=skiprows, usecols=usecols, unpack=unpack,
                 ndmin=ndmin, encoding=encoding, max_rows=max_rows, quote=quotechar)


def savetxt(fname, X, fmt="%.18e", delimiter=" ", newline="\n", header="", footer="",
            comments="# ", encoding=None):
    if hasattr(fname, "__fspath__"):
        fname = os.fspath(fname)
    if not isinstance(fname, str) and not hasattr(fname, "write"):
        raise ValueError("fname must be a string or file handle")
    X = np.asarray(X)
    if X.ndim == 0 or X.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D array instead")
    if X.ndim == 1:
        X = np.atleast_2d(X).T
    ncol = X.shape[1]
    iscomplex_X = np.iscomplexobj(X)
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError(f"fmt has wrong shape.  {str(fmt)}")
        row_format = delimiter.join(fmt)
    elif isinstance(fmt, str):
        n_fmt_chars = fmt.count("%")
        error = ValueError(f"fmt has wrong number of % formats:  {fmt}")
        if n_fmt_chars == 1:
            if iscomplex_X:
                fmt = [f" ({fmt}+{fmt}j)"] * ncol
            else:
                fmt = [fmt] * ncol
            row_format = delimiter.join(fmt)
        elif iscomplex_X and n_fmt_chars != 2 * ncol:
            raise error
        elif not iscomplex_X and n_fmt_chars != ncol:
            raise error
        else:
            row_format = fmt
    else:
        raise ValueError(f"invalid fmt: {fmt!r}")
    # Build the text first: each write to a simulated file rewrites the whole file.
    pieces = []
    if len(header) > 0:
        header = header.replace("\n", "\n" + comments)
        pieces.append(comments + header + newline)
    for row in X.tolist():
        if iscomplex_X:
            parts = []
            for number in row:
                parts.extend((number.real, number.imag))
            pieces.append((row_format % tuple(parts) + newline).replace("+-", "-"))
            continue
        try:
            pieces.append(row_format % tuple(row) + newline)
        except TypeError as e:
            raise TypeError(
                f"Mismatch between array dtype ('{str(X.dtype)}') and "
                f"format specifier ('{row_format}')"
            ) from e
    if len(footer) > 0:
        footer = footer.replace("\n", "\n" + comments)
        pieces.append(comments + footer + newline)
    text = "".join(pieces)
    if isinstance(fname, str):
        with open(fname, "wt", encoding=encoding) as handle:
            handle.write(text)
        return
    try:
        fname.write(text)
    except TypeError:
        fname.write(text.encode(encoding or "latin1"))


class _LineSplitter:
    """genfromtxt's field splitter: comments are cut first, then the line is split at a
    delimiter string, whitespace, or fixed field widths."""

    def __init__(self, delimiter=None, comments="#", autostrip=True, encoding=None):
        self.comments = _decode_line(comments)
        self.encoding = encoding
        self.autostrip = autostrip
        delimiter = _decode_line(delimiter)
        if delimiter is None or isinstance(delimiter, str):
            self.delimiter = delimiter or None
            self.split = self._delimited
        elif hasattr(delimiter, "__iter__"):
            edges = [0]
            for width in delimiter:
                edges.append(edges[-1] + width)
            self.delimiter = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            self.split = self._variable_width
        elif int(delimiter):
            self.delimiter = int(delimiter)
            self.split = self._fixed_width
        else:
            self.delimiter = None
            self.split = self._delimited

    def _strip_comment(self, line):
        if self.comments is not None:
            line = line.split(self.comments)[0]
        return line

    def _delimited(self, line):
        line = self._strip_comment(line).strip(" \r\n")
        if not line:
            return []
        return line.split(self.delimiter)

    def _fixed_width(self, line):
        line = self._strip_comment(line).strip("\r\n")
        if not line:
            return []
        width = self.delimiter
        return [line[i:i + width] for i in range(0, len(line), width)]

    def _variable_width(self, line):
        line = self._strip_comment(line)
        if not line:
            return []
        return [line[start:end] for start, end in self.delimiter]

    def __call__(self, line):
        fields = self.split(_decode_line(line, self.encoding))
        if self.autostrip:
            return [field.strip() for field in fields]
        return fields


def _str2bool(value):
    value = value.upper()
    if value == "TRUE":
        return True
    if value == "FALSE":
        return False
    raise ValueError("Invalid boolean")


# genfromtxt's type ladder for ``dtype=None``: each column takes the first entry that parses
# all of its values, and missing values fill with the entry's default.
_LADDER = [
    (np.dtype(bool), _str2bool, False),
    (np.dtype(np.int64), int, -1),
    (np.dtype(np.float64), float, np.nan),
    (np.dtype(np.complex128), complex, complex(np.nan, 0)),
    (np.dtype(str), str, "???"),
]


def _converter_for(dtype):
    """The conversion function and default fill value NumPy's ``StringConverter`` uses for a
    given dtype."""
    kind = dtype.kind
    if kind == "b":
        return _str2bool, False
    if kind in "iu":
        if dtype == np.dtype(np.uint64):
            return np.uint64, -1
        if dtype == np.dtype(np.int64):
            return np.int64, -1
        return (lambda text: int(float(text))), -1
    if kind == "f":
        return float, np.nan
    if kind == "c":
        return complex, complex(np.nan, 0)
    return str, "???"


def _fits_int64(text):
    value = int(text)
    if not -(2**63) <= value < 2**63:
        raise ValueError(text)
    return value


def _infer_column(values, missing):
    """The ladder entry NumPy's ``StringConverter.iterupgrade`` settles on for a column."""
    for level, (dtype, convert, default) in enumerate(_LADDER):
        if level == 1:
            convert = _fits_int64
        try:
            for value in values:
                try:
                    convert(value)
                except ValueError:
                    if value.strip() not in missing:
                        raise
        except ValueError:
            continue
        return _LADDER[level]
    return _LADDER[-1]


def genfromtxt(fname, dtype=float, comments="#", delimiter=None, skip_header=0, skip_footer=0,
               converters=None, missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None, deletechars=None, replace_space="_",
               autostrip=False, case_sensitive=True, defaultfmt="f%i", unpack=None,
               usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding=None,
               *, ndmin=0, like=None):
    _ensure_ndmin_ndarray_check_param(ndmin)
    if max_rows is not None:
        if skip_footer:
            raise ValueError(
                "The keywords 'skip_footer' and 'max_rows' can not be specified at the same time."
            )
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")
    if names is not None and names is not False:
        raise NotImplementedError(
            "genfromtxt with names= builds structured arrays, which shellsim's NumPy does not support"
        )
    if usemask:
        raise NotImplementedError("masked arrays are not supported by shellsim's NumPy")
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError(
            "The input argument 'converter' should be a valid dictionary "
            f"(got '{type(user_converters)}' instead)"
        )
    if encoding == "bytes":
        encoding = None
    if dtype is not None:
        dtype = np.dtype(dtype)
    lines = _source_lines(fname, encoding)
    split_line = _LineSplitter(delimiter=delimiter, comments=comments, autostrip=autostrip,
                               encoding=encoding)
    position = skip_header
    first_values = []
    while position < len(lines) and not first_values:
        first_line = lines[position]
        first_values = split_line(first_line)
        position += 1
    if not first_values:
        warnings.warn(f'genfromtxt: Empty input file: "{fname}"', stacklevel=2)
        first_line = ""
        position = len(lines)
    body = [first_line] + lines[position:] if first_values else []

    if usecols is not None:
        if isinstance(usecols, str):
            usecols = [part.strip() for part in usecols.split(",")]
        else:
            try:
                usecols = list(usecols)
            except TypeError:
                usecols = [usecols]
        usecols = [column + len(first_values) if column < 0 else column for column in usecols]
    nbcols = len(usecols or first_values)

    user_missing = missing_values or ()
    if isinstance(user_missing, bytes):
        user_missing = user_missing.decode("latin1")
    missing = [[""] for _ in range(nbcols)]
    if isinstance(user_missing, dict):
        for key, value in user_missing.items():
            if usecols and key in usecols:
                key = usecols.index(key)
            values = [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
            targets = missing if key is None else [missing[key]]
            for entry in targets:
                entry.extend(values)
    elif isinstance(user_missing, (list, tuple)):
        for value, entry in zip(user_missing, missing):
            if str(value) not in entry:
                entry.append(str(value))
    elif isinstance(user_missing, str):
        for entry in missing:
            entry.extend(user_missing.split(","))
    else:
        for entry in missing:
            entry.append(str(user_missing))

    fills = [None] * nbcols
    if isinstance(filling_values, dict):
        for key, value in filling_values.items():
            if usecols and key in usecols:
                key = usecols.index(key)
            fills[key] = value
    elif isinstance(filling_values, (list, tuple)):
        n = len(filling_values)
        if n <= nbcols:
            fills[:n] = filling_values
        else:
            fills = list(filling_values[:nbcols])
    elif filling_values is not None:
        fills = [filling_values] * nbcols

    rows = []
    invalid = []
    for i, line in enumerate(body):
        values = split_line(line)
        nbvalues = len(values)
        if nbvalues == 0:
            continue
        if usecols:
            try:
                values = [values[column] for column in usecols]
            except IndexError:
                invalid.append((i + skip_header + 1, nbvalues))
                continue
        elif nbvalues != nbcols:
            invalid.append((i + skip_header + 1, nbvalues))
            continue
        rows.append(values)
        if len(rows) == max_rows:
            break

    if invalid:
        nbrows = len(rows) + len(invalid) - skip_footer
        if skip_footer > 0:
            skipped = len([entry for entry in invalid if entry[0] > nbrows + skip_header])
            invalid = invalid[:len(invalid) - skipped]
            skip_footer -= skipped
        if invalid:
            message = "\n".join(
                ["Some errors were detected !"]
                + [f"    Line #{line} (got {count} columns instead of {nbcols})"
                   for line, count in invalid]
            )
            if invalid_raise:
                raise ValueError(message)
            warnings.warn(message, ConversionWarning, stacklevel=2)
    if skip_footer > 0:
        rows = rows[:-skip_footer]

    columns = []
    column_dtypes = []
    for index in range(nbcols):
        texts = [row[index] for row in rows]
        if dtype is None:
            column_dtype, convert, default = _infer_column(texts, missing[index])
        else:
            column_dtype = dtype
            convert, default = _converter_for(dtype)
        if fills[index] is not None:
            default = fills[index]
        user = user_converters.get(index)
        if user is None and usecols:
            user = user_converters.get(usecols[index])
        if user is not None:
            convert = user
        converted = []
        for text in texts:
            try:
                converted.append(convert(text))
            except ValueError:
                if not loose and text.strip() not in missing[index]:
                    raise ValueError(f"Cannot convert string '{text}'") from None
                converted.append(default)
        columns.append(converted)
        column_dtypes.append(column_dtype)

    if dtype is None:
        kinds = {column_dtype.kind for column_dtype in column_dtypes}
        if len(kinds) > 1:
            raise NotImplementedError(
                "genfromtxt columns of different types need a structured array, "
                "which shellsim's NumPy does not support"
            )
        dtype = column_dtypes[0] if column_dtypes else np.dtype(np.float64)
    flat = [columns[c][r] for r in range(len(rows)) for c in range(nbcols)]
    output = np.array(flat, dtype=dtype)
    if rows:
        output = output.reshape(len(rows), nbcols)
    output = _ensure_ndmin_ndarray(output, ndmin=ndmin)
    if unpack:
        return output.T
    return output
