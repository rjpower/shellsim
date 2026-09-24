# Goal: build zlib inside shellsim

## Acceptance target

Starting from a pinned, unmodified upstream zlib source tree in shellsim's virtual filesystem,
run `./configure` and `make` through shellsim's shell. The build must produce a usable library and
executable test program, and the latter must run against virtual files and streams without any
ambient host filesystem, process, network, clock, or compiler access. Capture exact source revision,
toolchain inputs, output format, and resource limits in a deterministic integration test. A command
that cannot do the requested work must fail explicitly, not return success with a placeholder file.

The first reference is upstream zlib 1.3.2 (`v1.3.2`), whose release archive has SHA-256
`b99a0b86c0ba9360ec7e78c4f1e43b1cbdf1e6936c8fa0f6835c0cd694a495a1`. Its normal
`configure` probes `cc`, `ar`, `ranlib`, and `nm`; the generated Makefile compiles C objects,
archives `libz.a`, links examples, and also builds a shared library. `./configure --static` may be
used as an intermediate gate, but it does not satisfy the unqualified acceptance target.

## Work sequence

1. Add a deterministic source-loading harness and run the actual `./configure` and `make` commands
   in shellsim. Keep the first failures as checked, named integration cases.
2. Close shell and build-tool gaps exposed by the real scripts: executable script dispatch,
   substitutions, redirections, tests, text tools, Makefile expansion and graph evaluation. Fix
   coherent semantics rather than special-casing zlib; reject unsupported constructs visibly.
3. Choose and integrate a Wasm-hosted C toolchain that accepts zlib's C and conventional object,
   archive, and link workflow. WCPL currently proves only self-contained small C programs; its
   custom `.wo` format, limited preprocessor, and broken libc-backed output are not enough for
   unchanged zlib. Evaluate a Wasm-hosted Clang/LLD or another compact toolchain against measured
   C and resource requirements before committing to an object ABI.
4. Wire `cc`, `ar`, `ranlib`, and linker invocation through virtual process execution. Keep native
   Rust commands and Wasm executables on the same virtual syscall boundary. Any compiler package
   data or sysroot must live in the virtual filesystem, not be mounted from the host.
5. Run `./configure && make` and a zlib round-trip or upstream example entirely in shellsim;
   then add invalid-input and bounded-resource cases. Keep the full pipeline in CI at a practical
   cost, with smaller narrow cases for individual failures.

## Decision points and risks

- The current Wasm adapter buffers stdio at dispatch. Compiler probes often use sequential files,
  but pipeline-driven build recipes will require resumable Wasm I/O to avoid deadlocks.
- Upstream's default build includes shared-library linking. Wasm's module/linking model does not
  directly provide ELF `.so` semantics. We need either a compatible virtual shared-library model
  or an explicit target-compatible configuration path; do not silently omit `shared`.
- Building a full compiler inside Wasmi may exceed current memory, module-size, and CPU budgets.
  Increase them only with bounded accounting and measured fixtures, not unlimited host work.
- Do not execute upstream build scripts on the host as the product path. A reference build in an
  isolated test harness may aid diagnosis, but the acceptance result must come from shellsim.

## Current evidence

The integration harness extracts the pinned, unmodified zlib 1.3.2 source archive into the VFS.
Its PAX global `comment` record is accepted as inert metadata; other global PAX keys remain
explicitly unsupported. The pinned [XCC](https://github.com/tyfkda/xcc) guest, with a documented
compiler and libc patch, completes `./configure && make` and produces `libz.a`, `example`, and
`minigzip`. The `example` executable runs under shellsim and reports a successful compress and
uncompress round trip. All compiler inputs, intermediate files, archive members, and linked
executables stay inside the virtual filesystem. No host compiler or archive tool is invoked by
the product path.

The work exposed general gaps rather than zlib-specific source changes: PAX metadata handling,
Make rule expansion and continued recipes, an open-but-unlinked file lifetime, WASI filesystem
operations, backward `goto` lowering in XCC, missing C headers/libc stream error reporting, and
an indexed Wasm-object `ar` implementation. WASI Preview 1 cannot change file permission bits,
so the guest linker uses the explicit `shellsim.path_chmod` extension to mark an executable.

The current test sets a bounded 20-billion-unit CPU limit and 128 MiB VFS disk limit. It checks
the full default build, not only `make static`; zlib's configure selects its supported build mode
for this Wasm toolchain. This is not a claim that XCC supports arbitrary C projects or ELF shared
libraries. The default zlib build is the acceptance target; wider toolchain compatibility and
shared-library semantics remain separate work. The older
[Wasm-hosted Clang demonstration](https://github.com/binji/wasm-clang) uses a legacy WASI ABI and
large modules; modern [YoWASP Clang](https://github.com/YoWASP/clang) is more complete but its
published package uses split Wasm modules and JavaScript orchestration. Neither was needed to
reach this target.
