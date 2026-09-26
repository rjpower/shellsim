# Wasm execution spike: measured gates and compiler choice

## Result

Shellsim recognizes executable Wasm bytes in its VFS and runs bounded modules with Wasmtime 49.
A custom WASI Preview 1 adapter uses the process's virtual descriptors, virtual clock, exported process
environment, deterministic random source, and process-owned VFS descriptors. A separately
compiled Rust `wasm32-wasip1` `wc` fixture reads piped input and virtual files, and its selected
outputs match the native `wc`. Unsupported WASI calls trap if reached, and other import
namespaces fail instantiation; neither grants
the guest host filesystem, process, network, environment, or clock access. WAT integration cases
cover stdout, stdin, exit status, arguments, environment, time, regular-file creation, invalid
guest pointers, memory limits, fuel exhaustion, and unknown imports. A multi-stage `wc` pipeline
also handles input larger than pipe capacity under default limits. A Wasm guest runs as a
scheduled process that suspends on a live pipe; a snapshot of a running guest fails explicitly.

A concrete compiler trial used WCPL revision `458a542ca81fa7a8fd8c8eed38a32dce8ed45135` in a
temporary checkout. A trusted native bootstrap built a 307 KiB `wcpl.wasm`. Loaded into shellsim's
VFS, that Wasm-hosted compiler read `minimal.c`, emitted `minimal.wasm`, and shellsim executed the
result with status 0. This required about 9.6 million modeled CPU units under a 100-million-unit
test budget. A `puts` program found the virtual `<stdio.h>` after explicit `-I`/`-L` paths but the
Wasm-hosted compiler failed to load its `stdio.wo` object; the same source and library paths worked
in the native bootstrap. That is a concrete library-linking frontier, not evidence of zlib support.

An earlier pinned compiler was tested through a shell-level workflow:
`/usr/bin/wcpl` read virtual C files, wrote a virtual Wasm executable, and the shell executed
that output with the C program's exit status. A two-file C program linked in one invocation.
Invalid source and CPU exhaustion failed visibly. WCPL compiles and links in one invocation, so no
process-spawn extension is needed. The checked case is self-contained C, not a libc-backed program.
Further probing found that the Wasm-hosted compiler constructs
`res://lib/include\stdio.h` for its embedded header on this path. A local path-selection change
let it find the header, but a `puts` program
printed only its first character and a `putchar` program produced invalid Wasm. Those failures
remain a compiler-guest compatibility gate, not a supported workflow.

The first Wasmi build used a recursive instruction dispatcher and overflowed the host stack on a
100,000-fuel loop. Its `portable-dispatch` feature fixed that case before the Wasmtime migration.
Keep the loop regression test: fuel alone is insufficient if an engine consumes host stack per step.

## Stage gates

| Gate | Current state | Required next work |
|---|---|---|
| 0. Validate and run bounded Wasm with virtual stdio and VFS | Demonstrated with WAT ABI tests and a source-built Rust WASI `wc`. Regular-file handles use process-owned descriptors; a multi-stage `wc` pipeline handles input larger than pipe capacity. | Extend the compiled guest fixture to exercise file writes and close/reopen. |
| 1. Model a full logical Wasm process | **Done.** A Wasm executable is a scheduled process image. Blocked stream calls suspend the Wasmtime async stack on the scheduler's wait reason; fuel yields bound each quantum. A fork's copy of a running guest fails with status 126. Pipe, prompt, kill, timeout, SIGPIPE, and shared-CPU tests cover it. | Record and replay WASI responses if forks of running guests must continue. |
| 2. Build surface | **Partial.** A pinned external TinyCC package and wasi-libc subset compile and run C programs in the VFS. No compiler is installed by default. | Broaden the checked libc and process surface with real applications; keep unsupported calls explicit. |

The snapshot constraint is substantive. Shellsim's process continuations and environments are
cloneable, and a harness fork must make an independent replay snapshot. Wasmtime's live `Store` is
not cloneable. Putting it in an `Arc` would share mutable guest state between
forks and violate that contract. A coherent path is to record bounded WASI responses and replay
the guest deterministically to a suspension point when cloning; another is to make an active-Wasm
session explicitly non-forkable, which would narrow the current session API. Do not hide either
change behind a shallow wrapper. Scheduled guests currently take the second path: a fork succeeds, but its copy of a
running guest fails explicitly.

Archive work should not precede the object ABI. A Unix `ar` container without a compatible symbol
index is not enough for `ld`, and an archive of WCPL's `.wo` files differs from an archive of
standard relocatable Wasm objects. `ranlib` must build or validate a useful index, not silently
report success. The current `ar`/`ranlib` implementation handles indexed Wasm objects but does not
establish compatibility with tinycc's output.

## Compiler candidates

| Candidate | Fit | Constraint |
|---|---|---|
| [WCPL](https://github.com/false-schemers/wcpl) | An earlier C-subset experiment proved a small self-contained compile/run path. | Its custom `.wo` format and libc failures do not suit the unchanged zlib target. Its binary fixture was removed. |
| [TinyCC Wasm fork](https://github.com/rjpower/tinycc/tree/wasm) | Its pinned external package compiles and links C programs against a wasi-libc subset through shellsim. | The supported libc and WASI surface is still limited; unsupported imports trap if called. |
| [Chibicc](https://github.com/rui314/chibicc) | Broad C11 frontend and preprocessor; a possible frontend reference. | Emits x86-64 assembly and targets native Linux, so it still needs a Wasm backend and linker. The upstream repository may rewrite history. |
| [WASI SDK](https://github.com/WebAssembly/wasi-sdk) / [Wasm Clang demo](https://github.com/binji/wasm-clang) | Full Clang/LLVM C compatibility and standard Wasm object format. A Wasm-hosted Clang/LLD precedent exists. | Large runtime/sysroot footprint and integration cost; the demo has custom memory filesystem plumbing and describes itself as alpha. |
| [PunyCC](https://github.com/bobbl/punycc) | Tiny, self-hosted, and already has Wasm host and target combinations. | No preprocessor, linker, standard library, or useful type system. Good engine smoke test, not a zlib compiler. |

The next compiler gate is broader application behavior, not another WCPL or XCC fixture.
Wasmtime accepts TinyCC's exception instructions, and the pinned package plus sysroot exercises
the bounded virtual operations currently implemented. This does not establish full libc support.
