# Guest Clang compilation and loading

Shellsim compiles Wasm functions in parallel and persists Wasmtime native artifacts across host
processes. The process-local LRU also retains successful executable admission for exact original
Wasm bytes. Compilation CPU charges, scratch reservations, thread limits, and execution fuel
remain the same on cold and cached launches.

The native cache lives outside the VFS, under the host user's cache directory at
`shellsim/wasmtime` (`$XDG_CACHE_HOME/shellsim/wasmtime` or `$HOME/.cache/shellsim/wasmtime` on
Linux). The default is an 8 GiB soft disk limit. Wasmtime cleans it asynchronously; the limit is
not a hard quota. Its separate process-local LRU retains at most 64 modules and 8 GiB, including
source, compiled image, and a conservative estimate of retained admission metadata.

Host process settings are read when the first Wasm engine is created:

| Setting | Meaning |
| --- | --- |
| `SHELLSIM_WASMTIME_CACHE_DIR` | Absolute cache directory; `off` disables persistence |
| `SHELLSIM_WASMTIME_CACHE_CONFIG` | Host path to a Wasmtime cache TOML configuration |

A directory override takes precedence over the configuration file's directory. With a custom
configuration file, Wasmtime's defaults apply to omitted size/cleanup settings. For example:

```toml
[cache]
directory = "/absolute/private/wasmtime-cache"
files-total-size-soft-limit = "16Gi"
```

Unix cache directories must be private (mode 0700); Linux also checks the effective user owner.
A symlink at the cache root is rejected. Cache files contain trusted host executable code: use
host-owned storage protected from untrusted writers, including its parent directories. Guest
paths, guest environment variables, and guest-supplied serialized native code cannot select or
populate this cache. An unavailable or invalid cache reports a host diagnostic and leaves Wasm
compilation enabled.

Wasmtime keys artifacts by Wasm content, compiler settings, target ISA, and a custom module
version. Shellsim's version includes a SHA-256 identity over the patched Wasmtime sources, Cargo
manifest and lockfiles, and pinned Rust toolchain. This prevents different patched runtimes that
share upstream version 49.0.0 from reusing incompatible native code. Concurrent same-source
misses share a compilation lock; unrelated process-local cache hits keep their own access.
Native disk-cache hits still run admission once in each process. Repeated executable launches
can reuse the positive admission profile only from an exact original-byte LRU entry; side-module
and rewritten-startup entries do not supply an executable profile.

## Observations

These are single shared-host observations on x86_64 Linux with 16 logical CPUs, Rust 1.97.1,
patched Wasmtime 49.0.0, and optimized installed Python wheels. They are not statistical
estimates. The frozen guest Clang was 84,236,270 bytes with SHA-256
`c471102c52afc7f75ebd29f7249db8893894b7d401b7ebc60ceefafd4c2abbb2`; LLD SHA-256 was
`086d5eb07d2b99f2021b61303d4f2d0c4da7f22c5aeaadb725a2a38dfe94d3d0`.
Native LLVM and package build products were reused. Installation is measured separately from
Wasm loading and actual guest compilation/linking. The earlier public project phase took
437.36 seconds and included installation, `make -j2`, program execution, and Python package
operations.

| Phase | Previous cold process | Previous same process | Changed cold process | Changed same process |
| --- | ---: | ---: | ---: | ---: |
| Clang load and `--version` | 267.88 s | 3.29 s | 38.54 s | 0.260 s |
| LLD load and `--version` | Included in first link | Included in first link | 19.85 s | 0.085 s |
| Compile `main.c` | 3.53 s | 3.19 s | 0.285 s | 0.283 s |
| Compile `codec.c` | 3.71 s | 3.19 s | 0.291 s | 0.285 s |
| Link | 135.53 s, including cold LLD | 4.85 s | 0.570 s | 0.573 s |
| Execute linked program | 4.95 s | 0.077 s | 0.765 s | 0.066 s |

A fresh second host process reused the persistent artifacts: Clang load/version took 4.432
seconds, LLD 2.093 seconds, C compilation 0.281 seconds each, link 0.545 seconds, and execution
0.142 seconds. Release installation for that process took 7.02 seconds.

The original `make -j2` project took 7.741 seconds in a fresh host process with populated native
artifacts, and 0.642 seconds on the next same-process rebuild. Program execution took 0.132 and
0.065 seconds. The complete original public project passed in **31.523 seconds**, including
release installation, `make -j2`, execution, and SciPy/ImageIO/Kiwi operations. This was a fresh
host process with populated Clang/LLD/make/codec native artifacts; the earlier **437.363 seconds**
used a cold serial runtime without a persistent Wasmtime cache. The cold loading measurements
remain 38.54 seconds for Clang and 19.85 seconds for LLD, as recorded separately above.
[The checked measurement data](guest-clang-performance-results.json) preserves the commands,
source identity, setup duration, and observations.

The previous runtime had both Wasmtime disk caching and parallel compilation disabled. A host
admission probe measured approximately 0.81 seconds for the Clang raw atomic wait scan and
0.82 seconds for its thread-profile scan, with no executable startup rewrite. Reusing positive
admission in the process-local LRU removes those repeated passes after the same guest charges.

The standalone host compilation sweep kept fuel, shared memory, threads, EH, and the async
stack profile enabled. It excludes installation, admission scans, instantiation, and guest code.
Native DWARF `debug_info` was already false. The optional symbols/address-map change saved image
space without an observed compilation speed gain, so production keeps its diagnostics. Disabling
optimization was slower in this observation; production keeps Cranelift's `Speed` default.

| Compiler settings | Wasm compilation | Native image |
| --- | ---: | ---: |
| Cranelift Speed, symbols and address maps | 18.45 s | 270,777,592 bytes |
| Cranelift Speed, symbols/address maps disabled | 19.33 s | 231,433,864 bytes |
| Cranelift None, symbols and address maps | 26.72 s | 303,539,040 bytes |
| Winch, required fuel/thread/EH profile | Rejected Clang at function 27491: illegal fuel state | No artifact |

Winch could construct the required engine but failed during actual Clang compilation with fuel
enabled. Production retains Cranelift. No runtime capability was disabled for this comparison.
Allocation, memory reservation, stack size, shared-memory support, and fuel settings retain the
command engine's existing profile; changing them would require separate compatibility and
resource evidence. Wasmtime's [Config documentation](https://docs.rs/wasmtime/49.0.0/wasmtime/struct.Config.html)
describes the available settings; the vendored 49.0.0 APIs govern this build.

## Reproduction

Install an optimized shellsim wheel and use the same immutable Clang/zlib/make release for each
run. Set `runtime_python`, `release`, and `install_cache` to the installed interpreter, release
manifest, and existing release-installation cache. An empty private native cache measures cold
loading; keep it for subsequent host processes.

```sh
export SHELLSIM_WASMTIME_CACHE_DIR="$HOME/.cache/shellsim/clang-measurement"
uv run --no-project --python "$runtime_python" infra/guest-clang-performance.py \
  --release "$release" --cache "$install_cache" --output target/clang-first.json
uv run --no-project --python "$runtime_python" infra/guest-clang-performance.py \
  --release "$release" --cache "$install_cache" --repeat 1 --output target/clang-second.json
uv run --no-project --python "$runtime_python" infra/guest-clang-performance.py \
  --release "$release" --cache "$install_cache" --make --output target/clang-make.json
```

The script verifies the linked zlib program's output and records each command, host duration,
exit status, guest resource usage, stdout/stderr, and the Clang content identity. `--make`
rebuilds the original project with `make -j2` in each repetition.

The separate compilation sweep uses the original host-readable Wasm artifact:

```sh
cargo run --release --example wasmtime_compile --features wasmtime/winch -- \
  "$clang_wasm" cranelift speed parallel full
cargo run --release --example wasmtime_compile --features wasmtime/winch -- \
  "$clang_wasm" winch speed parallel full
```

Replace `full` with `minimal` to omit symbols/address maps, or `speed` with `none` to measure
unoptimized code generation. The opt-in admission probe reads the same artifact:

```sh
SHELLSIM_CLANG_FIXTURE="$clang_wasm" cargo test --lib measure_guest_admission_scans \
  -- --ignored --nocapture
```
