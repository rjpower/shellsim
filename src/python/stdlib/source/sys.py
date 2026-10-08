"""Interpreter metadata and process streams for the modeled Python runtime.

The values that depend on the running process (``argv``, the streams, the import path, the
active exception and the loaded modules) come from the ``_sys`` capability module. Everything
else describes the modeled 64-bit Linux CPython 3.14 build so that version and platform checks in
ordinary programs take their usual branches.
"""

from _sys import argv, path, stdin, stdout, stderr, exit, maxsize, executable, prefix
from _sys import _getframemodulename
import _sys

__stdin__ = stdin
__stdout__ = stdout
__stderr__ = stderr

platform = "linux"
byteorder = "little"
maxunicode = 0x10FFFF
exec_prefix = prefix
base_prefix = prefix
base_exec_prefix = prefix
platlibdir = "lib"
abiflags = ""
api_version = 1013
version = "3.14.0 (shellsim)"
hexversion = 0x030E00F0
copyright = "shellsim modeled Python runtime"
float_repr_style = "short"
dont_write_bytecode = True
pycache_prefix = None
warnoptions = []
meta_path = []
path_hooks = []
path_importer_cache = {}
orig_argv = [executable] + list(argv)
_git = ("shellsim", "", "")


def _struct_sequence(name, fields, repr_name=None):
    namespace = {"_fields": fields, "n_fields": len(fields), "n_sequence_fields": len(fields)}

    def make_property(index):
        return property(lambda self: self[index])

    for index, field in enumerate(fields):
        namespace[field] = make_property(index)

    def __new__(cls, *values):
        if len(values) == 1 and isinstance(values[0], (tuple, list)):
            values = tuple(values[0])
        return tuple.__new__(cls, values)

    def __repr__(self):
        return (repr_name or name) + "(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(fields, self)
        ) + ")"

    namespace["__new__"] = __new__
    namespace["__repr__"] = __repr__
    return type(name, (tuple,), namespace)


_version_info = _struct_sequence(
    "version_info", ("major", "minor", "micro", "releaselevel", "serial"), "sys.version_info"
)
version_info = _version_info(3, 14, 0, "final", 0)

_flags = _struct_sequence(
    "flags",
    (
        "debug",
        "inspect",
        "interactive",
        "optimize",
        "dont_write_bytecode",
        "no_user_site",
        "no_site",
        "ignore_environment",
        "verbose",
        "bytes_warning",
        "quiet",
        "hash_randomization",
        "isolated",
        "dev_mode",
        "utf8_mode",
        "warn_default_encoding",
        "safe_path",
        "int_max_str_digits",
        "gil",
        "thread_inherit_context",
        "context_aware_warnings",
    ),
    "sys.flags",
)
flags = _flags(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, False, 1, 0, False, 4300, 1, 0, 0)

_float_info = _struct_sequence(
    "float_info",
    (
        "max",
        "max_exp",
        "max_10_exp",
        "min",
        "min_exp",
        "min_10_exp",
        "dig",
        "mant_dig",
        "epsilon",
        "radix",
        "rounds",
    ),
    "sys.float_info",
)
float_info = _float_info(
    1.7976931348623157e308, 1024, 308, 2.2250738585072014e-308, -1021, -307, 15, 53,
    2.220446049250313e-16, 2, 1,
)

_int_info = _struct_sequence(
    "int_info",
    ("bits_per_digit", "sizeof_digit", "default_max_str_digits", "str_digits_check_threshold"),
    "sys.int_info",
)
int_info = _int_info(30, 4, 4300, 640)

_hash_info = _struct_sequence(
    "hash_info",
    ("width", "modulus", "inf", "nan", "imag", "algorithm", "hash_bits", "seed_bits", "cutoff"),
    "sys.hash_info",
)
hash_info = _hash_info(64, 2305843009213693951, 314159, 0, 1000003, "siphash13", 64, 128, 0)

_thread_info = _struct_sequence("thread_info", ("name", "lock", "version"), "sys.thread_info")
thread_info = _thread_info("pthread", "mutex+cond", None)


class _Implementation:
    def __init__(self):
        self.name = "shellsim"
        self.cache_tag = None
        self.version = version_info
        self.hexversion = hexversion
        self._multiarch = "x86_64-linux-gnu"
        self.supports_isolated_interpreters = False

    def __repr__(self):
        items = sorted(vars(self).items())
        return "namespace(" + ", ".join(key + "=" + repr(value) for key, value in items) + ")"


implementation = _Implementation()

builtin_module_names = (
    "_asyncio", "_collections", "_functools", "_io", "_json", "_operator", "_os", "_sre",
    "_struct", "_sys", "_time", "builtins", "errno", "itertools", "math", "sys", "time",
)
stdlib_module_names = frozenset(
    (
        "abc", "argparse", "asyncio", "base64", "binascii", "bisect", "builtins", "cmath", "codecs",
        "collections", "contextlib", "copy", "copyreg", "csv", "dataclasses", "datetime", "decimal", "enum", "errno",
        "fractions", "functools", "glob", "hashlib", "heapq", "html", "http", "importlib", "inspect",
        "io", "itertools", "json", "keyword", "logging", "math", "operator", "os", "pathlib", "pkgutil",
        "posixpath", "random", "re", "shutil", "signal", "stat", "statistics", "string",
        "struct", "subprocess", "sys", "tempfile", "textwrap", "threading", "time", "types", "typing",
        "unicodedata", "unittest", "urllib", "uuid", "warnings", "zipfile", "zlib",
    )
)


class _Modules:
    """Live view of the interpreter's module table, as ``sys.modules`` exposes it."""

    def __getitem__(self, name):
        return _sys.modules()[name]

    def __setitem__(self, name, module):
        _sys.set_module(name, module)

    def __delitem__(self, name):
        if not _sys.remove_module(name):
            raise KeyError(name)

    def __contains__(self, name):
        return name in _sys.modules()

    def __iter__(self):
        return iter(_sys.modules())

    def __len__(self):
        return len(_sys.modules())

    def get(self, name, default=None):
        return _sys.modules().get(name, default)

    def pop(self, name, *default):
        modules = _sys.modules()
        if name in modules:
            _sys.remove_module(name)
            return modules[name]
        if default:
            return default[0]
        raise KeyError(name)

    def setdefault(self, name, default=None):
        if name not in self:
            self[name] = default
        return self[name]

    def keys(self):
        return _sys.modules().keys()

    def values(self):
        return _sys.modules().values()

    def items(self):
        return _sys.modules().items()

    def copy(self):
        return _sys.modules()

    def __repr__(self):
        return repr(_sys.modules())


modules = _Modules()

_recursion_limit = 256
_trace = None
_profile = None
_switch_interval = 0.005
_int_max_str_digits = 4300
_dlopen_flags = 2
_coroutine_origin_tracking_depth = 0
_asyncgen_hooks = (None, None)


def exception():
    return _sys.active_exception()


def exc_info():
    active = _sys.active_exception()
    if active is None:
        return (None, None, None)
    return (type(active), active, getattr(active, "__traceback__", None))


def getrecursionlimit():
    return _recursion_limit


def setrecursionlimit(limit):
    global _recursion_limit
    if not isinstance(limit, int):
        raise TypeError("'%s' object cannot be interpreted as an integer" % type(limit).__name__)
    if limit < 1:
        raise ValueError("recursion limit must be greater or equal than 1")
    _recursion_limit = limit


def getrefcount(obj):
    return 2


def getsizeof(obj, default=None):
    """Approximate CPython 3.14 object sizes on a 64-bit build by type."""
    if obj is None or obj is True or obj is False:
        return 16 if obj is None else 28
    if isinstance(obj, int):
        magnitude = abs(obj)
        digits = 1
        while magnitude >= 1 << 30:
            magnitude >>= 30
            digits += 1
        return 28 + 4 * (digits - 1)
    if isinstance(obj, float):
        return 24
    if isinstance(obj, complex):
        return 32
    if isinstance(obj, str):
        if obj.isascii():
            return 41 + len(obj)
        return 58 + 2 * len(obj)
    if isinstance(obj, (bytes, bytearray)):
        return 33 + len(obj)
    if isinstance(obj, tuple):
        return 40 + 8 * len(obj)
    if isinstance(obj, list):
        return 56 + 8 * len(obj)
    if isinstance(obj, dict):
        return 64 + 32 * len(obj) if obj else 64
    if isinstance(obj, (set, frozenset)):
        return 216 + 16 * len(obj) if len(obj) > 4 else 216
    if default is not None:
        return default
    return 56


def intern(string):
    if not isinstance(string, str):
        raise TypeError("intern() argument must be str, not " + type(string).__name__)
    return string


def getdefaultencoding():
    return "utf-8"


def getfilesystemencoding():
    return "utf-8"


def getfilesystemencodeerrors():
    return "surrogateescape"


def get_int_max_str_digits():
    return _int_max_str_digits


def set_int_max_str_digits(maxdigits):
    if maxdigits != 0 and maxdigits < 640:
        raise ValueError("maxdigits must be 0 or larger than 640")
    if maxdigits != _int_max_str_digits:
        raise NotImplementedError("shellsim fixes int_max_str_digits at %d" % _int_max_str_digits)


def settrace(function):
    global _trace
    _trace = function


def gettrace():
    return _trace


def setprofile(function):
    global _profile
    _profile = function


def getprofile():
    return _profile


def getswitchinterval():
    return _switch_interval


def setswitchinterval(interval):
    global _switch_interval
    if interval <= 0:
        raise ValueError("switch interval must be strictly positive")
    _switch_interval = float(interval)


def getdlopenflags():
    return _dlopen_flags


def setdlopenflags(flags):
    global _dlopen_flags
    _dlopen_flags = flags


def get_coroutine_origin_tracking_depth():
    return _coroutine_origin_tracking_depth


def set_coroutine_origin_tracking_depth(depth):
    global _coroutine_origin_tracking_depth
    if depth < 0:
        raise ValueError("depth must be >= 0")
    _coroutine_origin_tracking_depth = depth


def get_asyncgen_hooks():
    return _asyncgen_hooks


def set_asyncgen_hooks(firstiter=None, finalizer=None):
    global _asyncgen_hooks
    _asyncgen_hooks = (firstiter, finalizer)


def audit(event, *args):
    return None


def addaudithook(hook):
    return None


def call_tracing(function, args):
    return function(*args)


def is_finalizing():
    return False


def is_stack_trampoline_active():
    return False


def is_remote_debug_enabled():
    return False


def getallocatedblocks():
    return 0


def getunicodeinternedsize(*, _only_immortal=False):
    return 0


def _getframe(depth=0):
    raise ValueError("call stack is not deep enough")


def displayhook(value):
    if value is None:
        return
    import builtins

    builtins._ = None
    stdout.write(repr(value) + "\n")
    builtins._ = value


def excepthook(exc_type, value, traceback):
    stderr.write(exc_type.__name__ + (": " + str(value) if str(value) else "") + "\n")


def unraisablehook(unraisable):
    stderr.write("Exception ignored in: " + repr(unraisable.object) + "\n")


def breakpointhook(*args, **kwargs):
    return None


__displayhook__ = displayhook
__excepthook__ = excepthook
__unraisablehook__ = unraisablehook
__breakpointhook__ = breakpointhook
