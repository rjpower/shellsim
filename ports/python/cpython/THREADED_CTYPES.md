# Threaded upstream ctypes

The threaded product compiles unchanged CPython 3.13.7 `_ctypes` sources against the separately admitted threaded libffi 3.5.2 provider. Both consume the exact normal LLVM/sysroot receipts, SDK frontend/config/resource receipt and generated CPython header/config receipt. The side extension declares `DT_NEEDED libffi.so`; the assembled bundle retains the package-free interpreter bytes and records both artifact identities.

Build with `python -m ports.python.cpython.stdlib_ctypes_threaded`, supplying `--runtime`, `--libffi`, `--sdk`, `--compiler`, `--overlay` and a fresh `--work` directory. Assemble with `python -m ports.python.cpython.threaded_ffi_assembly`, supplying the runtime, provider, extension and a fresh output. Use `uv run` for both commands.

The public acceptance selects `SHELLSIM_THREADED_CTYPES_BUNDLE` and an independently compiled `SHELLSIM_THREADED_CTYPES_CONSUMER`, then runs `tests/test_threaded_ctypes.py` beside this port. It uses real upstream `ctypes` and checks one shared callback in main and two Python threads, nested CDLL calls, per-thread TLS and errno, a foreign C pthread invoking Python through `PyGILState`, canonical `pythonapi`, retired callback slot nonreuse and unchanged interpreter bytes.

Primitive scalar signatures are supported. Aggregate and variadic signatures remain explicit frontiers. Callbacks cannot be published during loader initialization. The process retains at most 64 callback slots over its lifetime; released slots remain tombstones and are not reused. Static threaded FFI is not admitted by this dynamic ABI.
