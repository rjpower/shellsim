# Kiwi 1.5.1

`graph-recipe.json` builds the pinned upstream sdist through its actual PEP 517
setuptools backend for the threaded v3 cohort. Graph build dependencies supply
unchanged setuptools, setuptools_scm, packaging and CPPy wheels in a private
host import tree. CPPy's CppyBuildExt helper supplies its headers; upstream
setuptools_scm generates the version header. No replacement source inventory
or global host environment is required.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/kiwisolver/graph-recipe.json \
  --store /path/to/ports-cache \
  --output /path/to/kiwi-release --check
```

The public release installer accepts `kiwisolver==1.5.1`. Its guest probe covers
two-variable constraints, edit variables, corrected methods and properties,
invalid input, and translated C++ constraint errors. See
[backend admission](../../_support/PYTHON_BACKENDS.md) for the pinned offline
hook and target sysconfig contract. This GIL-enabled cohort does not enable
free-threaded Python or establish Matplotlib's other native providers.

## Historical dynamic v2 build

This port builds the seven upstream C++ sources as an independent CPython 3.13
extension for SDK 34 dynamic ABI v2. The fixed interpreter owns libc, libc++, the
C++ exception tag and Python runtime. The extension imports them and contains no
private runtime or external native provider.

The pinned CPPy wheel supplies headers only. The version header uses the upstream
SCM template and verified sdist metadata. Python files, package metadata and
licenses retain their upstream bytes. The recipe pins source, headers, target
configuration and build scripts; output records the actual SDK tool identities.

The source patch supplies the unused second argument required by CPython's
`PyCFunction` signature for all 17 `METH_NOARGS` methods and the closure argument
for its four property getters. Native platforms often
tolerate their original cast from one-argument functions; WebAssembly checks the
function type at each indirect call. Solver and exception implementations remain
unchanged. This GIL-enabled cohort does not enable free-threaded Python.

```sh
uv run --no-project --python 3.13 ports/python/kiwisolver/build.py \
  --archive /path/to/kiwisolver-1.5.1.tar.gz \
  --cppy-wheel /path/to/cppy-1.3.1-py3-none-any.whl \
  --cpython-source /path/to/Python-3.13.7 \
  --cpython-build /path/to/wasi-build \
  --sdk /path/to/wasi-sdk-34.0-x86_64-linux \
  --runtime /path/to/fixed-v2-bundle --output /tmp/kiwisolver
```

Add the emitted wheel to the existing private catalog's `packages` array with
name `kiwisolver`, version `1.5.1`, relative wheel path and SHA-256. Its native
manifest declares no provider dependencies. Install normally with
`runtime.install_pypi(env, 'kiwisolver==1.5.1')`.

The opt-in guest test uses `SHELLSIM_KIWISOLVER_BUNDLE`,
`SHELLSIM_KIWISOLVER_UNIVERSE` and `SHELLSIM_PATCHED_UV`. It checks a two-variable
solve, edit variables, corrected methods and getters, invalid input, and translated C++
constraint exceptions while the interpreter bytes remain unchanged. This port
does not establish support for Matplotlib's remaining font or contour providers.
