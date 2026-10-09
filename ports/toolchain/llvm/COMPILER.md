# Threaded WASI compiler

`compiler.py` builds the standard Clang driver, Clang resource headers, LLD and
LLVM tools from the pinned source and patches in `compiler-recipe.json`. It
builds the WebAssembly target. The accepted linker-only and llc products remain
separate immutable artifacts.

Run the producer with a verified source archive and explicit host tools:

```sh
uv run --no-project python -m ports.toolchain.llvm.compiler \
  --archive /path/to/llvm-project.tar.gz \
  --cc /path/to/host-cc --cxx /path/to/host-cxx \
  --cmake /path/to/cmake --ninja /path/to/ninja \
  --work /path/to/new-build-directory
```

The producer verifies source, patch and driver identities, uses four compile
jobs and one link job, and records host tool hashes. Each command is limited to
one hour and 12 GiB of address space. An existing output directory is rejected.
The manifest hashes the produced binaries and resource headers and records
aliases such as `clang++` and `wasm-ld`. Upstream licensing is retained.

Consumers select a verified threaded sysroot and explicit WASI target flags.
The backend option `-mllvm -wasm-enable-wasi-dynamic-tls` remains opt-in.
Standard preprocessing, compilation and linking are handled by Clang. Setup
may add a thin default-flag wrapper; it must not parse or rewrite compiler
invocations into a separate compilation pipeline.

Compiler tests exercise preprocessing, resource-header compilation and actual
TLS lowering. Cohort acceptance additionally compiles independent C/C++ Python
extensions and executes late TLS loading and typed exception behavior in the
threaded guest. The compiler artifact alone does not certify a runtime cohort.
