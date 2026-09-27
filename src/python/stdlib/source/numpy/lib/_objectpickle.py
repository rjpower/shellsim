"""NumPy array pickles without a general ``pickle`` module.

shellsim has no ``pickle``, but ``.npy`` files store object arrays as the protocol 4 pickle that
NumPy writes with ``pickle.dump``. ``dump`` writes the same bytes as CPython's C pickler for
arrays whose elements are None, bools, ints, floats, complex numbers, strings, bytes, NumPy
scalars and arrays, and lists, tuples and dicts of these, including its memo and frame layout.
Two strings or bytes objects that are equal share a memo entry, as interned literals do in
CPython. Other element types raise ``NotImplementedError``.

``load`` reads pickles of protocols 2 to 5, but resolves only the globals NumPy's array pickles
use: its array and scalar reconstructors, ``ndarray``, ``dtype``, and a few builtins. Reading a
file therefore never imports a module or calls arbitrary code; any other global raises
``UnpicklingError``, as a restricted ``pickle.Unpickler`` would.
"""

import struct

import numpy

_FRAME_SIZE_MIN = 4
_FRAME_SIZE_TARGET = 64 * 1024
_BATCHSIZE = 1000
_MULTIARRAY = "numpy._core.multiarray"


class UnpicklingError(Exception):
    pass


class _Global:
    """A module attribute that pickles as a reference, such as ``numpy.ndarray``."""

    def __init__(self, module, name):
        self.module = module
        self.name = name


class _Reduce:
    """An object that pickles as ``func(*args)``, then ``BUILD`` with ``state`` if given."""

    def __init__(self, key, func, args, state=None):
        self.key = key
        self.func = func
        self.args = args
        self.state = state


_RECONSTRUCT = _Global(_MULTIARRAY, "_reconstruct")
_SCALAR = _Global(_MULTIARRAY, "scalar")
_NDARRAY = _Global("numpy", "ndarray")
_DTYPE = _Global("numpy", "dtype")
_COMPLEX = _Global("builtins", "complex")


def _dtype_reduce(dtype):
    """``dtype.__reduce__()``: NumPy keeps one instance of each native builtin dtype, so equal
    native dtypes share a memo entry and other dtypes never do."""
    if dtype.kind == "O":
        descr, order, sizes, flags = "O8", "|", (-1, -1), 63
    elif dtype.kind == "U":
        descr, order, sizes, flags = f"U{dtype.itemsize // 4}", "<", (dtype.itemsize, 4), 8
    else:
        descr, sizes, flags = f"{dtype.kind}{dtype.itemsize}", (-1, -1), 0
        order = "|" if dtype.itemsize == 1 else dtype.str[0]
    key = ("dtype", dtype.str) if dtype.isnative else ("unique", object())
    state = (3, order, None, None, None, sizes[0], sizes[1], flags)
    return _Reduce(key, _DTYPE, (descr, False, True), state)


def _array_reduce(array):
    """``ndarray.__reduce__()``: object elements travel as a C-order list, others as bytes in
    the array's memory order."""
    fortran = bool(array.flags.fnc)
    if array.dtype.kind == "O":
        raw = array.ravel().tolist()
    else:
        raw = array.tobytes(order="A")
    state = (1, tuple(array.shape), _dtype_reduce(array.dtype), fortran, raw)
    return _Reduce(("id", id(array)), _RECONSTRUCT, (_NDARRAY, (0,), b"b"), state)


def _encode_long(value):
    """Two's-complement little-endian bytes of ``value``, as ``pickle.encode_long`` gives."""
    magnitude = -value - 1 if value < 0 else value
    bits = 0
    while magnitude:
        magnitude >>= 1
        bits += 1
    length = (bits >> 3) + 1
    remaining = value % (1 << (8 * length))
    data = bytearray()
    for _ in range(length):
        data.append(remaining & 0xFF)
        remaining >>= 8
    if value < 0 and length > 1 and data[-1] == 0xFF and data[-2] & 0x80:
        del data[-1]
    return bytes(data)


def _decode_long(data):
    value = 0
    for byte in reversed(data):
        value = (value << 8) | byte
    if data and data[-1] & 0x80:
        value -= 1 << (8 * len(data))
    return value


class _Pickler:
    def __init__(self):
        self.output = []
        self.framing = False
        self.frame = None
        self.frame_size = 0
        self.memo = {}
        # Objects memoized by id stay referenced so that no other object can reuse their id.
        self.alive = []

    def dump(self, obj):
        self.output.append(b"\x80\x04")
        self.framing = True
        self.save(obj)
        self.write(b".")
        self.commit_frame()

    def write(self, data):
        if not self.framing:
            self.output.append(data)
            return
        if self.frame is None:
            self.frame = []
            self.frame_size = 0
        self.frame.append(data)
        self.frame_size += len(data)

    def commit_frame(self):
        if self.frame is None:
            return
        data = b"".join(self.frame)
        if len(data) >= _FRAME_SIZE_MIN:
            self.output.append(b"\x95" + struct.pack("<Q", len(data)))
        self.output.append(data)
        self.frame = None

    def write_payload(self, header, payload):
        # Large payloads bypass framing so a reader can copy them without buffering.
        if len(payload) < _FRAME_SIZE_TARGET:
            self.write(header)
            self.write(payload)
            return
        self.commit_frame()
        self.output.append(header)
        self.output.append(payload)

    def memoize(self, key):
        self.memo[key] = len(self.memo)
        self.write(b"\x94")

    def memo_get(self, key):
        index = self.memo[key]
        if index < 256:
            self.write(b"h" + bytes([index]))
        else:
            self.write(b"j" + struct.pack("<I", index))

    def save(self, obj):
        # CPython's C pickler ends a frame at the start of the save that finds it full.
        if self.frame is not None and self.frame_size >= _FRAME_SIZE_TARGET:
            self.commit_frame()
        kind = type(obj)
        if obj is None:
            self.write(b"N")
        elif kind is bool:
            self.write(b"\x88" if obj else b"\x89")
        elif kind is int:
            self.save_int(obj)
        elif kind is float:
            self.write(b"G" + struct.pack(">d", obj))
        elif kind is str:
            self.save_memoized(("str", obj), self.save_str, obj)
        elif kind is bytes:
            self.save_memoized(("bytes", obj), self.save_bytes, obj)
        elif kind is tuple:
            self.save_memoized(("id", id(obj)), self.save_tuple, obj)
        elif kind is list:
            self.save_memoized(("id", id(obj)), self.save_list, obj)
        elif kind is dict:
            self.save_memoized(("id", id(obj)), self.save_dict, obj)
        elif kind is complex:
            reduced = _Reduce(("id", id(obj)), _COMPLEX, (obj.real, obj.imag))
            self.save_memoized(reduced.key, self.save_reduce, reduced, obj)
        elif kind is _Global:
            self.save_memoized(("global", obj.module, obj.name), self.save_global, obj)
        elif kind is _Reduce:
            self.save_memoized(obj.key, self.save_reduce, obj)
        elif isinstance(obj, numpy.ndarray):
            reduced = _array_reduce(obj)
            self.save_memoized(reduced.key, self.save_reduce, reduced, obj)
        elif isinstance(obj, numpy.generic):
            reduced = _Reduce(
                ("unique", object()),
                _SCALAR,
                (_dtype_reduce(obj.dtype), numpy.array(obj).tobytes()),
            )
            self.save_reduce(reduced)
        else:
            raise NotImplementedError(
                f"np.save cannot pickle {kind.__name__!r} array elements in shellsim"
            )

    def save_memoized(self, key, save, obj, keep=None):
        if key in self.memo:
            self.memo_get(key)
            return
        self.alive.append(obj if keep is None else keep)
        save(obj)

    def save_int(self, value):
        if -0x80000000 <= value <= 0x7FFFFFFF:
            if 0 <= value <= 0xFF:
                self.write(b"K" + bytes([value]))
            elif 0 <= value <= 0xFFFF:
                self.write(b"M" + struct.pack("<H", value))
            else:
                self.write(b"J" + struct.pack("<i", value))
            return
        data = _encode_long(value)
        if len(data) < 256:
            self.write(b"\x8a" + bytes([len(data)]) + data)
        else:
            self.write(b"\x8b" + struct.pack("<i", len(data)) + data)

    def save_str(self, value):
        data = value.encode("utf-8")
        if len(data) < 256:
            header = b"\x8c" + bytes([len(data)])
        elif len(data) <= 0xFFFFFFFF:
            header = b"X" + struct.pack("<I", len(data))
        else:
            header = b"\x8d" + struct.pack("<Q", len(data))
        self.write_payload(header, data)
        self.memoize(("str", value))

    def save_bytes(self, value):
        if len(value) < 256:
            header = b"C" + bytes([len(value)])
        elif len(value) <= 0xFFFFFFFF:
            header = b"B" + struct.pack("<I", len(value))
        else:
            header = b"\x8e" + struct.pack("<Q", len(value))
        self.write_payload(header, value)
        self.memoize(("bytes", value))

    def save_tuple(self, value):
        key = ("id", id(value))
        if not value:
            self.write(b")")
            return
        if len(value) <= 3:
            for item in value:
                self.save(item)
            if key in self.memo:
                # The tuple contains itself through a mutable element.
                self.write(b"0" * len(value))
                self.memo_get(key)
                return
            self.write(bytes([0x84 + len(value)]))
        else:
            self.write(b"(")
            for item in value:
                self.save(item)
            if key in self.memo:
                self.write(b"1")
                self.memo_get(key)
                return
            self.write(b"t")
        self.memoize(key)

    def save_list(self, value):
        self.write(b"]")
        self.memoize(("id", id(value)))
        if len(value) == 1:
            self.save(value[0])
            self.write(b"a")
            return
        start = 0
        while start < len(value):
            self.write(b"(")
            for item in value[start:start + _BATCHSIZE]:
                self.save(item)
            self.write(b"e")
            start += _BATCHSIZE

    def save_dict(self, value):
        self.write(b"}")
        self.memoize(("id", id(value)))
        items = list(value.items())
        if len(items) == 1:
            self.save(items[0][0])
            self.save(items[0][1])
            self.write(b"s")
            return
        start = 0
        while start < len(items):
            self.write(b"(")
            for key, item in items[start:start + _BATCHSIZE]:
                self.save(key)
                self.save(item)
            self.write(b"u")
            start += _BATCHSIZE

    def save_global(self, value):
        self.save(value.module)
        self.save(value.name)
        self.write(b"\x93")
        self.memoize(("global", value.module, value.name))

    def save_reduce(self, value):
        self.save(value.func)
        self.save(value.args)
        self.write(b"R")
        if value.key in self.memo:
            self.write(b"0")
            self.memo_get(value.key)
        else:
            self.memoize(value.key)
        if value.state is not None:
            self.save(value.state)
            self.write(b"b")


def dump(obj, file):
    """Write ``obj`` to ``file`` as ``pickle.dump(obj, file, protocol=4)`` does."""
    pickler = _Pickler()
    pickler.dump(obj)
    file.write(b"".join(pickler.output))


class _Pending:
    """An array or dtype under construction: the result of ``REDUCE``, completed by ``BUILD``.

    The memo slots that hold it are recorded so ``BUILD`` can replace them with the result.
    """

    def __init__(self, kind, args):
        self.kind = kind
        self.args = args
        self.slots = []


def _codecs_encode(text, encoding="utf-8"):
    return text.encode(encoding)


def _scalar(dtype, data):
    if dtype.kind == "O":
        return data
    return numpy.frombuffer(data, dtype=dtype)[0]


def _reconstruct(cls, shape, typecode):
    return _Pending("array", ())


_ALLOWED_GLOBALS = {
    (_MULTIARRAY, "_reconstruct"): _reconstruct,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    (_MULTIARRAY, "scalar"): _scalar,
    ("numpy.core.multiarray", "scalar"): _scalar,
    ("numpy", "ndarray"): numpy.ndarray,
    ("numpy", "dtype"): numpy.dtype,
    ("builtins", "complex"): complex,
    ("__builtin__", "complex"): complex,
    ("builtins", "set"): set,
    ("__builtin__", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("__builtin__", "frozenset"): frozenset,
    ("builtins", "bytearray"): bytearray,
    ("__builtin__", "bytearray"): bytearray,
    ("_codecs", "encode"): _codecs_encode,
}

_CALLABLE = (_reconstruct, _scalar, complex, set, frozenset, bytearray, _codecs_encode)


def _finish_dtype(pending, state):
    descr = pending.args[0]
    if not isinstance(descr, str):
        raise UnpicklingError("invalid dtype descriptor in pickle")
    if len(state) >= 5 and (state[3] is not None or state[4] is not None):
        raise NotImplementedError("structured dtypes are not supported by shellsim's NumPy")
    dtype = numpy.dtype(descr)
    if len(state) >= 2 and state[1] == ">":
        dtype = dtype.newbyteorder(">")
    return dtype


def _finish_array(state):
    if len(state) == 5:
        state = state[1:]
    if len(state) != 4:
        raise UnpicklingError("invalid ndarray state in pickle")
    shape, dtype, fortran, raw = state
    shape = tuple(shape)
    count = 1
    for length in shape:
        count *= length
    if dtype.kind == "O":
        if not isinstance(raw, list) or len(raw) != count:
            raise ValueError("object pickle not returning list")
        array = numpy.empty(count, dtype=object)
        for index, item in enumerate(raw):
            array[index] = item
        array = array.reshape(shape)
        return numpy.asfortranarray(array) if fortran else array
    if isinstance(raw, str):
        raw = raw.encode("latin1")
    if len(raw) != count * dtype.itemsize:
        raise ValueError("buffer size does not match array size")
    array = numpy.frombuffer(raw, dtype=dtype, count=count).copy()
    if fortran:
        return array.reshape(shape[::-1]).T
    return array.reshape(shape)


class _Unpickler:
    def __init__(self, data, encoding):
        self.data = data
        self.position = 0
        self.encoding = encoding
        self.stack = []
        self.marks = []
        self.memo = {}

    def take(self, count):
        end = self.position + count
        if end > len(self.data):
            raise UnpicklingError("pickle data was truncated")
        value = self.data[self.position:end]
        self.position = end
        return value

    def pop_mark(self):
        if not self.marks:
            raise UnpicklingError("could not find MARK")
        items = self.stack
        self.stack = self.marks.pop()
        return items

    def put(self, index):
        value = self.stack[-1]
        self.memo[index] = value
        if isinstance(value, _Pending):
            value.slots.append(index)

    def get(self, index):
        if index not in self.memo:
            raise UnpicklingError(f"Memo value not found at index {index}")
        self.stack.append(self.memo[index])

    def text(self, data):
        if self.encoding == "bytes":
            return data
        return data.decode(self.encoding)

    def find_class(self, module, name):
        value = _ALLOWED_GLOBALS.get((module, name))
        if value is None:
            raise UnpicklingError(f"global '{module}.{name}' is forbidden")
        return value

    def reduce(self, func, args):
        if func is numpy.dtype:
            return _Pending("dtype", args)
        if not any(func is allowed for allowed in _CALLABLE):
            raise UnpicklingError("only NumPy array and scalar reconstructors may be called")
        return func(*args)

    def build(self, state):
        pending = self.stack[-1]
        if not isinstance(pending, _Pending):
            raise UnpicklingError("BUILD is only supported for NumPy arrays and dtypes")
        if pending.kind == "dtype":
            value = _finish_dtype(pending, state)
        else:
            value = _finish_array(state)
        self.stack[-1] = value
        for index in pending.slots:
            self.memo[index] = value

    def load(self):
        stack = self.stack
        while True:
            opcode = self.take(1)[0]
            if opcode == 0x80:
                protocol = self.take(1)[0]
                if protocol > 5:
                    raise ValueError(f"unsupported pickle protocol: {protocol}")
            elif opcode == 0x95:
                self.take(8)
            elif opcode == 0x2E:
                if not stack:
                    raise UnpicklingError("unpickling stack underflow")
                value = stack.pop()
                if isinstance(value, _Pending):
                    raise UnpicklingError("pickle ends before its array is built")
                return value
            elif opcode == 0x28:
                self.marks.append(stack)
                self.stack = stack = []
            elif opcode == 0x30:
                if stack:
                    stack.pop()
                else:
                    self.pop_mark()
                    stack = self.stack
            elif opcode == 0x31:
                self.pop_mark()
                stack = self.stack
            elif opcode == 0x32:
                stack.append(stack[-1])
            elif opcode == 0x4E:
                stack.append(None)
            elif opcode == 0x88:
                stack.append(True)
            elif opcode == 0x89:
                stack.append(False)
            elif opcode == 0x4B:
                stack.append(self.take(1)[0])
            elif opcode == 0x4D:
                stack.append(struct.unpack("<H", self.take(2))[0])
            elif opcode == 0x4A:
                stack.append(struct.unpack("<i", self.take(4))[0])
            elif opcode == 0x8A:
                stack.append(_decode_long(self.take(self.take(1)[0])))
            elif opcode == 0x8B:
                length = struct.unpack("<i", self.take(4))[0]
                if length < 0:
                    raise UnpicklingError("LONG pickle has negative byte count")
                stack.append(_decode_long(self.take(length)))
            elif opcode == 0x47:
                stack.append(struct.unpack(">d", self.take(8))[0])
            elif opcode == 0x8C:
                stack.append(self.take(self.take(1)[0]).decode("utf-8"))
            elif opcode == 0x58:
                stack.append(self.take(struct.unpack("<I", self.take(4))[0]).decode("utf-8"))
            elif opcode == 0x8D:
                stack.append(self.take(struct.unpack("<Q", self.take(8))[0]).decode("utf-8"))
            elif opcode == 0x43:
                stack.append(self.take(self.take(1)[0]))
            elif opcode == 0x42:
                stack.append(self.take(struct.unpack("<I", self.take(4))[0]))
            elif opcode == 0x8E:
                stack.append(self.take(struct.unpack("<Q", self.take(8))[0]))
            elif opcode == 0x96:
                stack.append(bytearray(self.take(struct.unpack("<Q", self.take(8))[0])))
            elif opcode == 0x55:
                stack.append(self.text(self.take(self.take(1)[0])))
            elif opcode == 0x54:
                length = struct.unpack("<i", self.take(4))[0]
                if length < 0:
                    raise UnpicklingError("BINSTRING pickle has negative byte count")
                stack.append(self.text(self.take(length)))
            elif opcode == 0x29:
                stack.append(())
            elif opcode == 0x74:
                items = self.pop_mark()
                stack = self.stack
                stack.append(tuple(items))
            elif 0x85 <= opcode <= 0x87:
                count = opcode - 0x84
                if len(stack) < count:
                    raise UnpicklingError("unpickling stack underflow")
                items = tuple(stack[-count:])
                del stack[-count:]
                stack.append(items)
            elif opcode == 0x5D:
                stack.append([])
            elif opcode == 0x6C:
                items = self.pop_mark()
                stack = self.stack
                stack.append(list(items))
            elif opcode == 0x61:
                item = stack.pop()
                stack[-1].append(item)
            elif opcode == 0x65:
                items = self.pop_mark()
                stack = self.stack
                stack[-1].extend(items)
            elif opcode == 0x7D:
                stack.append({})
            elif opcode == 0x64:
                items = self.pop_mark()
                stack = self.stack
                stack.append({items[i]: items[i + 1] for i in range(0, len(items), 2)})
            elif opcode == 0x73:
                item = stack.pop()
                key = stack.pop()
                stack[-1][key] = item
            elif opcode == 0x75:
                items = self.pop_mark()
                stack = self.stack
                target = stack[-1]
                for i in range(0, len(items), 2):
                    target[items[i]] = items[i + 1]
            elif opcode == 0x8F:
                stack.append(set())
            elif opcode == 0x90:
                items = self.pop_mark()
                stack = self.stack
                stack[-1].update(items)
            elif opcode == 0x91:
                items = self.pop_mark()
                stack = self.stack
                stack.append(frozenset(items))
            elif opcode == 0x71:
                self.put(self.take(1)[0])
            elif opcode == 0x72:
                self.put(struct.unpack("<I", self.take(4))[0])
            elif opcode == 0x94:
                self.put(len(self.memo))
            elif opcode == 0x68:
                self.get(self.take(1)[0])
            elif opcode == 0x6A:
                self.get(struct.unpack("<I", self.take(4))[0])
            elif opcode == 0x63:
                module = self.readline().decode("ascii")
                name = self.readline().decode("ascii")
                stack.append(self.find_class(module, name))
            elif opcode == 0x93:
                name = stack.pop()
                module = stack.pop()
                if type(name) is not str or type(module) is not str:
                    raise UnpicklingError("STACK_GLOBAL requires str")
                stack.append(self.find_class(module, name))
            elif opcode == 0x52:
                args = stack.pop()
                func = stack.pop()
                stack.append(self.reduce(func, args))
            elif opcode == 0x62:
                self.build(stack.pop())
            else:
                raise UnpicklingError(f"unsupported pickle opcode 0x{opcode:02x}")

    def readline(self):
        end = self.data.find(b"\n", self.position)
        if end < 0:
            raise UnpicklingError("pickle data was truncated")
        value = self.data[self.position:end]
        self.position = end + 1
        return value


def load(file, encoding="ASCII"):
    """Read one pickle from ``file``, leaving it positioned after the pickle's ``STOP``."""
    start = file.tell()
    data = file.read()
    unpickler = _Unpickler(data, encoding)
    value = unpickler.load()
    file.seek(start + unpickler.position)
    return value
