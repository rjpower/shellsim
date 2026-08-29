# Compilers & build tools in shellsim — survey and decision

The brief asked whether to introduce a C compiler (tinycc / clang) or build-system support, or
whether that's too big a sandboxing burden. Short answer: **we do not embed a compiler, and that
is the correct call.** This note records why, and what we do instead.

## The core conflict: compilation implies native execution

A C/C++/Rust task is only "solved" when the produced binary **runs** and its behavior is checked.
That means the value isn't in `cc -o foo foo.c` succeeding — it's in executing `./foo`. But
executing arbitrary freshly-emitted native code is exactly the thing the sandbox boundary exists
to prevent:

- The seccomp filter (`src/sandbox.rs`) denies `execve`/`execveat` (plus `fork`/`vfork`, socket
  syscalls, `ptrace`). A real toolchain that shelled out to `gcc`/`ld`/`as` would be blocked at
  the first `execve`, and *should* be — those subprocesses are uninstrumented native code with
  full ambient authority.
- Even if we embedded an in-process compiler (tinycc can emit to memory; cranelift/LLVM via a
  crate), we would then have to **run** the emitted machine code in our address space. That is a
  direct, deliberate sandbox escape: JIT'd code is not subject to seccomp's syscall *table*
  rewriting in any cooperative way, and it bypasses every Python/shell-level guard we built. It
  would undo the environment's in-process, capability-limited execution model.

So a compiler buys us nothing we can safely use: compile-without-run is pointless for grading, and
compile-and-run is the one capability we are explicitly hardening against.

## What we do instead: honest OOD, not a fake success

`gcc`, `cc`, `clang`, `g++`, `c++`, `make`, `cmake`, `ninja`, `cargo build`, `go build`, etc. are
registered as `Trust::NoOp` — they "succeed" (exit 0, no error spew) so a Dockerfile/solve script
that merely *mentions* them doesn't abort, **but they produce no artifact.** The moment the task
tries to run the artifact (`./a.out`, `./target/release/foo`, …) it hits the
command-not-found / unsupported path, which records the capability gap. A compiled-language
workload is therefore flagged rather than silently treated as native execution.

## `make` specifically

A `Makefile` is a dependency DAG whose recipes are shell commands. Two cases:

- **Recipes that compile** (`$(CC) ...`) — dead-end as above; correctly OOD.
- **Recipes that are pure shell** (codegen, file shuffling, running a Python script, `tar`, `sed`)
  — these we *already* simulate, so a minimal `make` that parsed targets/prereqs and ran recipes
  through the existing shell would genuinely extend coverage for non-C projects.

We deliberately scoped that out for this round: it's a self-contained future addition (a recipe
runner over the shell we already have) with no sandboxing risk, but it wasn't on the critical path
to the scientific-Python coverage that the target dataset actually needs. Tracked as a follow-up.

## Decision summary

| Option | Verdict | Reason |
|---|---|---|
| Embed tinycc / cranelift / LLVM and **run** output | ✗ rejected | Direct sandbox escape; defeats seccomp |
| Embed a compiler, compile only (no run) | ✗ rejected | No grading value; cost without benefit |
| Treat compilers/build tools as `NoOp`, record capability gaps | ✓ **shipped** | Honest detection; zero new attack surface |
| Minimal shell-recipe `make` (no compilation) | ↪ future | Safe, useful for non-C Makefiles; out of scope this round |
