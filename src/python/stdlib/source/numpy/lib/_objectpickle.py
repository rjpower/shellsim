"""A clean-room protocol-4 pickler and restricted unpickler for `numpy.save`/`numpy.load`.

shellsim has no `pickle` module, so `np.save` needs its own encoder for `dtype=object`
arrays, and `np.load(allow_pickle=True)` needs a decoder that never imports a module or
calls arbitrary code. Both are built directly from the pickle protocol (PEP 3154) and the
`pickletools` opcode descriptions, and their output was checked byte for byte against
CPython 3.14's `pickle.dumps` and NumPy 2.5.3's `np.save` for the element types shellsim
supports: `None`, `bool`, `int`, `float`, `complex`, `str`, `bytes`, `list`, `tuple`, `dict`,
and NumPy scalars and arrays, nested to any depth.

Framing (protocol 4's `FRAME` opcode) is reproduced with CPython's own thresholds: pickled
bytes are buffered and flushed as a frame once the buffer reaches 64 KiB, and a `str`/`bytes`
payload at or above that size is written directly to the stream, forcing a flush of whatever
was buffered first. Memoization matches CPython's `MEMOIZE` scheme (one shared, sequential
index space) with one deliberate difference, noted in docs/numpy.md: shellsim strings have no
identity, so they are memoized by value. Every other memoized type (bytes, tuples, lists,
dicts, and the dtype/global objects built along the way) is memoized by object identity, as
CPython does.

The unpickler resolves only the small set of globals the pickler itself can produce
(`numpy._core.multiarray._reconstruct` and `.scalar`, `numpy.ndarray`, `numpy.dtype`, and
`builtins.complex`); anything else raises `UnpicklingError` before it is looked up or called,
so a crafted file cannot import a module or invoke arbitrary code.
"""

import struct

import numpy as np

_FRAME_SIZE_TARGET = 64 * 1024
_FRAME_SIZE_MIN = 4
_BATCH_SIZE = 1000

# `_reconstruct`'s third argument is always this exact one-byte object: NumPy passes the same
# shared `b"b"` constant for every array in a pickle (it lives in `ndarray.__reduce__`'s
# compiled code, reused on every call), so bytes memoization-by-identity picks it up as a
# repeat after the first array. A literal written fresh at each `_save_ndarray` call would be a
# distinct object every time and never get memoized, so it is hoisted to a single shared
# instance here.
_RECONSTRUCT_MODE = b"b"


class UnpicklingError(Exception):
    """A pickle stream named a disallowed global, was malformed, or ran out of bytes."""


def dumps(array):
    """The protocol-4 pickle bytes for a `numpy.ndarray`, matching CPython's `pickle.dumps`."""
    return _Pickler().dumps(array)


def loads(data):
    """Reconstruct the value a restricted `dumps`-compatible pickle stream encodes."""
    return _Unpickler(data).load()


def _pack_be_double(value):
    # BINFLOAT is the one big-endian field in an otherwise little-endian protocol.
    return struct.pack(">d", value)


def _unpack_be_double(data):
    return struct.unpack(">d", bytes(data))[0]


def _long_to_bytes(value):
    """Minimal little-endian two's-complement bytes for an arbitrary-precision int."""
    if value == 0:
        return b""
    magnitude = ~value if value < 0 else value
    return value.to_bytes(magnitude.bit_length() // 8 + 1, "little", signed=True)


def _long_from_bytes(data):
    return int.from_bytes(data, "little", signed=True)


class _Pickler:
    def __init__(self):
        self._out = bytearray()
        self._frame = bytearray()
        self._next_memo = 0
        self._id_memo = {}
        self._str_memo = {}
        self._global_memo = {}
        self._dtype_memo = {}

    def dumps(self, value):
        self._out += b"\x80\x04"
        self._save(value)
        self._frame += b"."
        self._commit(force=True)
        return bytes(self._out)

    # -- framing --------------------------------------------------------

    def _commit(self, force=False):
        if not self._frame:
            return
        if not force and len(self._frame) < _FRAME_SIZE_TARGET:
            return
        if len(self._frame) >= _FRAME_SIZE_MIN:
            self._out += b"\x95" + len(self._frame).to_bytes(8, "little")
        self._out += self._frame
        self._frame = bytearray()

    def _emit(self, data):
        self._frame += data

    def _emit_large(self, data):
        self._commit(force=True)
        self._out += data

    def _memoize(self):
        index = self._next_memo
        self._next_memo += 1
        self._emit(b"\x94")
        return index

    def _emit_get(self, index):
        if index < 256:
            self._emit(b"h" + bytes([index]))
        else:
            self._emit(b"j" + index.to_bytes(4, "little"))

    # -- dispatch ---------------------------------------------------------

    def _save(self, value):
        self._commit(force=False)
        if value is None:
            self._emit(b"N")
        elif value is True:
            self._emit(b"\x88")
        elif value is False:
            self._emit(b"\x89")
        elif value is np.ndarray:
            self._save_global("numpy", "ndarray")
        elif isinstance(value, np.dtype):
            self._save_dtype(value)
        elif isinstance(value, np.ndarray):
            self._save_ndarray(value)
        elif isinstance(value, np.generic):
            # NumPy scalars always pickle through `numpy._core.multiarray.scalar`, even the
            # ones (`float64`/`complex128`/`str_`/`bytes_`) that also subclass a builtin type
            # `isinstance` would otherwise match below, so this check must come first.
            self._save_scalar(value)
        elif isinstance(value, int):
            self._save_int(value)
        elif isinstance(value, float):
            self._emit(b"G" + _pack_be_double(value))
        elif isinstance(value, complex):
            self._save_complex(value)
        elif isinstance(value, str):
            self._save_str(value)
        elif isinstance(value, bytes):
            self._save_bytes(value)
        elif isinstance(value, tuple):
            self._save_tuple(value)
        elif isinstance(value, list):
            self._save_list(value)
        elif isinstance(value, dict):
            self._save_dict(value)
        else:
            raise NotImplementedError(
                f"np.save cannot pickle {type(value).__name__!r} array elements in shellsim"
            )

    def _save_int(self, value):
        if 0 <= value <= 0xFF:
            self._emit(b"K" + bytes([value]))
        elif 0x100 <= value <= 0xFFFF:
            self._emit(b"M" + value.to_bytes(2, "little"))
        elif -(2**31) <= value <= 2**31 - 1:
            self._emit(b"J" + value.to_bytes(4, "little", signed=True))
        else:
            body = _long_to_bytes(value)
            if len(body) < 256:
                self._emit(b"\x8a" + bytes([len(body)]) + body)
            else:
                self._emit(b"\x8b" + len(body).to_bytes(4, "little") + body)

    def _save_complex(self, value):
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        self._save_global("builtins", "complex")
        self._save_tuple((value.real, value.imag))
        self._emit(b"R")
        self._id_memo[id(value)] = self._memoize()

    def _save_str(self, value):
        cached = self._str_memo.get(value)
        if cached is not None:
            self._emit_get(cached)
            return
        encoded = value.encode("utf-8")
        opcode = _text_length_opcode(b"\x8c", b"X", b"\x8d", encoded)
        if len(encoded) >= _FRAME_SIZE_TARGET:
            self._emit_large(opcode)
        else:
            self._emit(opcode)
        self._str_memo[value] = self._memoize()

    def _save_bytes(self, value):
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        opcode = _text_length_opcode(b"C", b"B", b"\x8e", value)
        if len(value) >= _FRAME_SIZE_TARGET:
            self._emit_large(opcode)
        else:
            self._emit(opcode)
        self._id_memo[id(value)] = self._memoize()

    def _save_tuple(self, value):
        if not value:
            self._emit(b")")
            return
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        batched = len(value) > 3
        if batched:
            self._emit(b"(")
        for item in value:
            self._save(item)
        self._emit({1: b"\x85", 2: b"\x86", 3: b"\x87"}.get(len(value), b"t"))
        self._id_memo[id(value)] = self._memoize()

    def _save_list(self, value):
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        self._emit(b"]")
        self._id_memo[id(value)] = self._memoize()
        # A single-element list uses the plain APPEND opcode; every other size, including a
        # trailing remainder chunk of exactly 1 item after a full batch, still uses the
        # MARK-delimited APPENDS form (confirmed against the reference for sizes 1, 2, 999-1002,
        # 2000-2001).
        if len(value) == 1:
            self._save(value[0])
            self._emit(b"a")
            return
        for start in range(0, len(value), _BATCH_SIZE):
            chunk = value[start : start + _BATCH_SIZE]
            self._emit(b"(")
            for item in chunk:
                self._save(item)
            self._emit(b"e")

    def _save_dict(self, value):
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        self._emit(b"}")
        self._id_memo[id(value)] = self._memoize()
        items = list(value.items())
        # Same rule as `_save_list`: only a single-pair dict gets the plain SETITEM opcode.
        if len(items) == 1:
            key, val = items[0]
            self._save(key)
            self._save(val)
            self._emit(b"s")
            return
        for start in range(0, len(items), _BATCH_SIZE):
            chunk = items[start : start + _BATCH_SIZE]
            self._emit(b"(")
            for key, val in chunk:
                self._save(key)
                self._save(val)
            self._emit(b"u")

    def _save_global(self, module, name):
        key = (module, name)
        cached = self._global_memo.get(key)
        if cached is not None:
            self._emit_get(cached)
            return
        self._save_str(module)
        self._save_str(name)
        self._emit(b"\x93")
        self._global_memo[key] = self._memoize()

    def _save_dtype(self, dtype):
        key = dtype.str
        cached = self._dtype_memo.get(key)
        if cached is not None:
            self._emit_get(cached)
            return
        self._save_global("numpy", "dtype")
        code, byteorder, itemsize, alignment, flags = _dtype_pickle_fields(dtype)
        self._save_tuple((code, False, True))
        self._emit(b"R")
        self._dtype_memo[key] = self._memoize()
        self._save_tuple((3, byteorder, None, None, None, itemsize, alignment, flags))
        self._emit(b"b")

    def _save_scalar(self, value):
        cached = self._id_memo.get(id(value))
        if cached is not None:
            self._emit_get(cached)
            return
        self._save_global("numpy._core.multiarray", "scalar")
        dtype = value.dtype
        payload = np.array(value, dtype=dtype).tobytes()
        self._save_tuple((dtype, payload))
        self._emit(b"R")
        self._id_memo[id(value)] = self._memoize()

    def _save_ndarray(self, array):
        cached = self._id_memo.get(id(array))
        if cached is not None:
            self._emit_get(cached)
            return
        self._save_global("numpy._core.multiarray", "_reconstruct")
        self._save_tuple((np.ndarray, (0,), _RECONSTRUCT_MODE))
        self._emit(b"R")
        self._id_memo[id(array)] = self._memoize()
        fortran_order = bool(array.flags.f_contiguous) and not bool(array.flags.c_contiguous)
        order = "F" if fortran_order else "C"
        if array.dtype == np.dtype(object):
            data = array.reshape(-1, order=order).tolist()
        else:
            data = array.tobytes(order=order)
        self._save_tuple((1, array.shape, array.dtype, fortran_order, data))
        self._emit(b"b")


def _text_length_opcode(short_op, medium_op, long_op, payload):
    length = len(payload)
    if length < 256:
        return short_op + bytes([length]) + payload
    if length < 2**32:
        return medium_op + length.to_bytes(4, "little") + payload
    return long_op + length.to_bytes(8, "little") + payload


def _dtype_pickle_fields(dtype):
    """(code, byteorder, itemsize, alignment, flags) for a dtype's pickled `__reduce__` state.

    Matches NumPy's own encoding: the code is the dtype's `.str` without its byte-order
    character, except for `object`, which NumPy pickles as the legacy code `'O8'`. Plain
    numeric dtypes carry no real itemsize/alignment/flags in the state (`-1, -1, 0`); `str`
    dtypes carry their true byte itemsize, a 4-byte alignment, and flag `8`; `object` carries
    `-1, -1, 63`. These constants were read off `pickle.dumps(np.dtype(...))` for every dtype
    shellsim supports.
    """
    if dtype.kind == "O":
        return "O8", "|", -1, -1, 63
    if dtype.kind == "U":
        return f"U{dtype.itemsize // 4}", "<", dtype.itemsize, 4, 8
    return dtype.str[1:], dtype.str[0], -1, -1, 0


# -- restricted unpickling ------------------------------------------------

_RECONSTRUCT = object()
_NDARRAY_CLASS = object()
_DTYPE_CTOR = object()
_SCALAR_CTOR = object()

_ALLOWED_GLOBALS = {
    ("numpy._core.multiarray", "_reconstruct"): _RECONSTRUCT,
    ("numpy", "ndarray"): _NDARRAY_CLASS,
    ("numpy", "dtype"): _DTYPE_CTOR,
    ("numpy._core.multiarray", "scalar"): _SCALAR_CTOR,
    ("builtins", "complex"): complex,
}


def _resolve_global(module, name):
    key = (module, name)
    if key not in _ALLOWED_GLOBALS:
        raise UnpicklingError(f"global '{module}.{name}' is forbidden")
    return _ALLOWED_GLOBALS[key]


class _Pending:
    """A reduced-but-not-yet-built object: the target of a later `BUILD` opcode."""

    __slots__ = ("kind", "payload", "memo_index")

    def __init__(self, kind, payload=None):
        self.kind = kind
        self.payload = payload
        self.memo_index = None


class _Reader:
    def __init__(self, data):
        self._data = data
        self._pos = 0

    def read(self, count):
        end = self._pos + count
        if end > len(self._data):
            raise UnpicklingError("pickle data was truncated")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def read_line(self):
        newline = self._data.find(b"\n", self._pos)
        if newline < 0:
            raise UnpicklingError("pickle data was truncated")
        line = self._data[self._pos : newline]
        self._pos = newline + 1
        return line


class _Unpickler:
    def __init__(self, data):
        self._reader = _Reader(bytes(data))
        self._stack = []
        self._marks = []
        self._memo = []

    def load(self):
        while True:
            op = self._reader.read(1)
            if op == b".":
                if not self._stack:
                    raise UnpicklingError("pickle data was truncated")
                return self._stack[-1]
            self._dispatch(op)

    def _dispatch(self, op):
        reader = self._reader
        stack = self._stack
        if op == b"\x80":
            reader.read(1)
        elif op == b"\x95":
            reader.read(8)
        elif op == b"N":
            stack.append(None)
        elif op == b"\x88":
            stack.append(True)
        elif op == b"\x89":
            stack.append(False)
        elif op == b"K":
            stack.append(reader.read(1)[0])
        elif op == b"M":
            stack.append(int.from_bytes(reader.read(2), "little"))
        elif op == b"J":
            stack.append(int.from_bytes(reader.read(4), "little", signed=True))
        elif op == b"\x8a":
            length = reader.read(1)[0]
            stack.append(_long_from_bytes(reader.read(length)))
        elif op == b"\x8b":
            length = int.from_bytes(reader.read(4), "little")
            stack.append(_long_from_bytes(reader.read(length)))
        elif op == b"G":
            stack.append(_unpack_be_double(reader.read(8)))
        elif op == b"\x8c":
            length = reader.read(1)[0]
            stack.append(reader.read(length).decode("utf-8"))
        elif op == b"X":
            length = int.from_bytes(reader.read(4), "little")
            stack.append(reader.read(length).decode("utf-8"))
        elif op == b"\x8d":
            length = int.from_bytes(reader.read(8), "little")
            stack.append(reader.read(length).decode("utf-8"))
        elif op == b"C":
            length = reader.read(1)[0]
            stack.append(bytes(reader.read(length)))
        elif op == b"B":
            length = int.from_bytes(reader.read(4), "little")
            stack.append(bytes(reader.read(length)))
        elif op == b"\x8e":
            length = int.from_bytes(reader.read(8), "little")
            stack.append(bytes(reader.read(length)))
        elif op == b")":
            stack.append(())
        elif op == b"]":
            stack.append([])
        elif op == b"}":
            stack.append({})
        elif op == b"(":
            self._marks.append(len(stack))
        elif op in (b"\x85", b"\x86", b"\x87"):
            count = {b"\x85": 1, b"\x86": 2, b"\x87": 3}[op]
            items = tuple(stack[len(stack) - count :])
            del stack[len(stack) - count :]
            stack.append(items)
        elif op == b"t":
            mark = self._marks.pop()
            items = tuple(stack[mark:])
            del stack[mark:]
            stack.append(items)
        elif op == b"a":
            value = stack.pop()
            stack[-1].append(value)
        elif op == b"e":
            mark = self._marks.pop()
            items = stack[mark:]
            del stack[mark:]
            stack[-1].extend(items)
        elif op == b"s":
            value = stack.pop()
            key = stack.pop()
            stack[-1][key] = value
        elif op == b"u":
            mark = self._marks.pop()
            items = stack[mark:]
            del stack[mark:]
            target = stack[-1]
            for index in range(0, len(items), 2):
                target[items[index]] = items[index + 1]
        elif op == b"\x94":
            top = stack[-1]
            if isinstance(top, _Pending):
                top.memo_index = len(self._memo)
            self._memo.append(top)
        elif op == b"h":
            self._push_memo(reader.read(1)[0])
        elif op == b"j":
            self._push_memo(int.from_bytes(reader.read(4), "little"))
        elif op == b"c":
            module = reader.read_line().decode("utf-8")
            name = reader.read_line().decode("utf-8")
            stack.append(_resolve_global(module, name))
        elif op == b"\x93":
            name = stack.pop()
            module = stack.pop()
            stack.append(_resolve_global(module, name))
        elif op == b"R":
            self._op_reduce()
        elif op == b"b":
            self._op_build()
        else:
            raise UnpicklingError(f"unsupported pickle opcode {op!r}")

    def _push_memo(self, index):
        if index >= len(self._memo):
            raise UnpicklingError("pickle data was truncated")
        self._stack.append(self._memo[index])

    def _op_reduce(self):
        args = self._stack.pop()
        target = self._stack.pop()
        if target is _RECONSTRUCT:
            result = _Pending("array")
        elif target is _DTYPE_CTOR:
            result = _Pending("dtype", args[0])
        elif target is _SCALAR_CTOR:
            dtype, payload = args
            result = np.frombuffer(payload, dtype=dtype)[0]
        elif target is complex:
            result = complex(*args)
        else:
            raise UnpicklingError("only NumPy array and scalar reconstructors may be called")
        self._stack.append(result)

    def _op_build(self):
        state = self._stack.pop()
        obj = self._stack.pop()
        if not isinstance(obj, _Pending):
            raise UnpicklingError("only NumPy array and scalar reconstructors may be called")
        if obj.kind == "dtype":
            final = _reconstruct_dtype(obj.payload, state)
        else:
            final = _reconstruct_array(state)
        if obj.memo_index is not None:
            self._memo[obj.memo_index] = final
        self._stack.append(final)


def _reconstruct_dtype(code, state):
    _version, byteorder, _subarray, _names, _fields, _itemsize, _alignment, _flags = state
    if code.startswith("O"):
        return np.dtype(object)
    if code.startswith("U"):
        return np.dtype("<U" + code[1:])
    order = byteorder if byteorder in ("<", ">") else "<"
    return np.dtype(order + code)


def _reconstruct_array(state):
    _version, shape, dtype, fortran_order, data = state
    order = "F" if fortran_order else "C"
    if dtype == np.dtype(object):
        # `np.array(data, dtype=object)` would try to nest same-length elements (a tuple
        # and a list of matching length look like two rows of a 2-d array to it); filling
        # a pre-shaped empty array element by element keeps `data` a flat list of objects.
        array = np.empty(len(data), dtype=object)
        for index, value in enumerate(data):
            array[index] = value
    else:
        count = 1
        for dim in shape:
            count *= dim
        array = np.frombuffer(data, dtype=dtype, count=count).copy()
    if shape:
        return array.reshape(shape, order=order)
    return array.reshape(())
