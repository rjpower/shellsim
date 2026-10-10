# Guest Clang and Wasm LLD

`guest-recipe.json` builds upstream Clang, Wasm LLD and LLVM archive tools as
WASI executables. Shellsim runs those commands inside the guest. The separate
`host-recipe.json` supplies the native compiler and TableGen programs used to
build them; installing a guest tool never exposes a host executable.

The guest compiler uses the admitted SDK 34 threaded libc and C++ runtime,
with LLVM's own worker threads disabled. Its WebAssembly backend retains the
reviewed TLS and linker patches used by the host compiler. Clang handles normal
preprocessing, C and C++ compilation, response files and driver dispatch. Linking
calls the guest `wasm-ld` through the virtual `posix_spawn` and `waitpid` facade.
The executable and the programs it emits use `wasm32-wasip1-threads`.

The install dependency `guest-sdk-recipe.json` supplies headers, libc++, libc,
compiler builtins and resource headers under `/usr/local/wasi-sysroot` and
`/usr/local/lib/clang/23`. These development files are installed data, rather
than native host build tools. Adjacent `clang.cfg` and `clang++.cfg` select the
guest defaults. Explicit Clang options retain their ordinary driver meaning.
The defaults link the genuine SDK `libunwind` for C++ exceptions and a small
startup object carrying the accepted `shellsim.abi` custom section. That object
is assembled from pinned source by the graph's admitted host compiler. Generated
programs can execute directly without host postprocessing. Commands and configs
install together under `/usr/bin`; `cc`, `c++`, `ar` and `ranlib` are guest shell
wrappers for the corresponding LLVM commands.

`guest.py` consumes the graph's verified archive, host compiler receipt, SDK,
platform sysroot and POSIX archive. It reuses the receipt's native `llvm-tblgen`,
`llvm-min-tblgen` and `clang-tblgen`; it does not build another host LLVM.
The locked persistent Ninja workspace retains source, dependency snapshots,
logs and interrupted objects. Each retained source is checked against the
pinned archive and exact patches before a build. Changed inputs require a
new workspace when source, generators or headers change. Verified library
updates replace archives at stable paths and let Ninja relink existing objects.
Installed commands are stripped with the admitted host `llvm-strip`, preserving
the main TLS classification in `dylink.0` and the `shellsim.abi` section.
Completed command products are immutable and verified before
being copied into the graph's private staging directory.

The producer honors the requested compilation job count, up to eight jobs,
and uses one link job. Build commands have
a one-hour timeout and a 12 GiB address-space limit per process. The guest
compiler executable reserves a 16 MiB C stack, starts with 128 MiB of linear
memory and permits growth to 256 MiB, subject to Shellsim's memory budget.

Wasmtime keeps compiled commands in a process-wide in-memory LRU cache, shared
by environments in the same host process. On 64-bit hosts it retains up to
64 modules and 8 GiB of source plus compiled images, allocating only as entries
arrive. Guest resource accounting is the same for a cache hit and a miss.
This cache is separate from the retained Ninja workspace and Cargo's on-disk
build cache; its size limit does not apply to those build outputs.

This port does not enable native LLVM backends, the optional static analyzer,
Objective-C rewriting or other LLD executable formats. Unix sockets, file
ownership changes, disk-capacity queries, child memory limits, detached launch,
timed child waits and watchdog timers reject unsupported requests. Guest traps
terminate the virtual process; asynchronous signal callbacks and Unix crash
recovery are disabled. Resource usage is not replaced with host or wall-clock
statistics. SDK anonymous memory mapping remains available; unsupported file
mapping uses LLVM's existing file-read fallback.

The development artifact preserves the SDK's empty `include/c++/v1` directory.
Upstream Clang uses that directory to discover the libc++ version before adding
the target's `eh/c++/v1` or `noeh/c++/v1` header directory.
