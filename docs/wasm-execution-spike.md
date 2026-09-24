# Wasm execution spike: measured gates and compiler choice

## Result

Shellsim now recognizes executable Wasm bytes in its VFS and runs bounded modules with Wasmi 2.0.
A custom WASI Preview 1 adapter uses shellsim's buffered streams, virtual clock, exported process
environment, deterministic random source, and process-owned VFS descriptors. A separately
compiled Rust `wasm32-wasip1` `wc` fixture reads piped input and virtual files, and its selected
outputs match the native `wc`. It rejects unknown imports and never grants
the guest host filesystem, process, network, environment, or clock access. WAT integration cases
cover stdout, stdin, exit status, arguments, environment, time, regular-file creation, invalid
guest pointers, memory limits, fuel exhaustion, and unknown imports. A multi-stage `wc` pipeline
also handles input larger than pipe capacity under default limits. Stdio is still buffered at
command dispatch; a Wasm guest cannot yet suspend on a live pipe or be cloned as a live process
snapshot.

A concrete compiler trial used WCPL revision `458a542ca81fa7a8fd8c8eed38a32dce8ed45135` in a
temporary checkout. A trusted native bootstrap built a 307 KiB `wcpl.wasm`. Loaded into shellsim's
VFS, that Wasm-hosted compiler read `minimal.c`, emitted `minimal.wasm`, and shellsim executed the
result with status 0. This required about 9.6 million modeled CPU units under a 100-million-unit
test budget. A `puts` program found the virtual `<stdio.h>` after explicit `-I`/`-L` paths but the
Wasm-hosted compiler failed to load its `stdio.wo` object; the same source and library paths worked
in the native bootstrap. That is a concrete library-linking frontier, not evidence of zlib support.

The pinned compiler is now checked in under `guest/wcpl/` and covered by a shell-level workflow:
`/usr/bin/wcpl` reads virtual C files, writes a virtual Wasm executable, and the shell executes
that output with the C program's exit status. A two-file C program links in one invocation.
Invalid source and CPU exhaustion fail visibly. WCPL compiles and links in one invocation, so no
process-spawn extension is needed. The checked case is self-contained C, not a libc-backed program.
Further probing found that the Wasm-hosted compiler constructs
`res://lib/include\stdio.h` for its embedded header on this path. A local path-selection change
let it find the header, but a `puts` program
printed only its first character and a `putchar` program produced invalid Wasm. Those failures
remain a compiler-guest compatibility gate, not a supported workflow.

The first Wasmi build used a recursive instruction dispatcher and overflowed the host stack on a
100,000-fuel loop. Enabling Wasmi's `portable-dispatch` feature fixed that case. Keep the loop
regression test: fuel alone is insufficient if the interpreter consumes host stack per step.

## Stage gates

| Gate | Current state | Required next work |
|---|---|---|
| 0. Validate and run bounded Wasm with virtual stdio and VFS | Demonstrated with WAT ABI tests, a standalone Rust WASI `wc`, and a Wasm-hosted compiler workflow. Regular-file handles use process-owned descriptors; a multi-stage `wc` pipeline handles input larger than pipe capacity. | Extend the compiled guest fixture to exercise file writes and close/reopen. |
| 1. Model a full logical Wasm process | **Open.** Buffered stdin works; live pipe reads cannot suspend and resume a Wasm instruction. | Use Wasmi's resumable host-trap API at shellsim's scheduler boundary. Decide how live continuations interact with deterministic session forks. Add pipe, cancellation, and timeout tests. |
| 2. Build surface | **Partial beyond the checked zlib workflow.** A limited Wasm C toolchain and virtual archive/link commands build pinned zlib through `./configure && make`; see [the build result](zlib-build-goal.md). | Broaden compiler and object compatibility before claiming general C package support. |

The snapshot constraint is substantive. Shellsim's process continuations and environments are
cloneable, and a harness fork must make an independent replay snapshot. Wasmi's live `Store` and
resumable call are not cloneable. Putting them in an `Arc` would share mutable guest state between
forks and violate that contract. A coherent path is to record bounded WASI responses and replay
the guest deterministically to a suspension point when cloning; another is to make an active-Wasm
session explicitly non-forkable, which would narrow the current session API. Do not hide either
change behind a shallow wrapper.

Archive work should not precede the object ABI. A Unix `ar` container without a compatible symbol
index is not enough for `ld`, and an archive of WCPL's `.wo` files differs from an archive of
standard relocatable Wasm objects. `ranlib` must build or validate a useful index, not silently
report success. The current implementation leaves these commands explicitly unsupported.

## Compiler candidates

| Candidate | Fit | Constraint |
|---|---|---|
| [WCPL](https://github.com/false-schemers/wcpl) | Best next experiment: MIT-licensed, self-hosted C-subset compiler, linker, and libc already targeting Wasm/WASI. Its README shows `wcpl.wasm` recompiling itself. | Uses a custom extended-WAT `.wo` object format and lacks a full C preprocessor and newer C features. An unchanged zlib build is unlikely without porting work. |
| [TinyCC](https://github.com/TinyCC/tinycc) | Stronger conventional C frontend and existing archive/linker machinery. | No Wasm target in the listed architectures. The backend, WASI bootstrap, object/linker path, and LGPL distribution review are substantial work. |
| [Chibicc](https://github.com/rui314/chibicc) | Broad C11 frontend and preprocessor; a possible frontend reference. | Emits x86-64 assembly and targets native Linux, so it still needs a Wasm backend and linker. The upstream repository may rewrite history. |
| [WASI SDK](https://github.com/WebAssembly/wasi-sdk) / [Wasm Clang demo](https://github.com/binji/wasm-clang) | Full Clang/LLVM C compatibility and standard Wasm object format. A Wasm-hosted Clang/LLD precedent exists. | Large runtime/sysroot footprint and integration cost; the demo has custom memory filesystem plumbing and describes itself as alpha. |
| [PunyCC](https://github.com/bobbl/punycc) | Tiny, self-hosted, and already has Wasm host and target combinations. | No preprocessor, linker, standard library, or useful type system. Good engine smoke test, not a zlib compiler. |

The new evidence changes the proposed compiler order: try WCPL under the virtual WASI adapter
before forking TinyCC. Pin a WCPL revision, build its Wasm-hosted compiler with trusted external
tooling, load it as a fixture, and test whether it can compile and run a small multi-file program
entirely in shellsim. If its C or `.wo` boundary blocks the target workload, a TinyCC backend or
Wasm-hosted Clang/LLD can be judged against a concrete gap. Do not claim zlib support until the
actual checked build runs, links, and executes its output in the virtual environment.
