# shellsim

`shellsim` is a deterministic, resource-constrained BusyBox-like environment for evaluating
agents. Shell programs and Unix-style commands run in-process against an in-memory filesystem;
they never execute host programs or use the host filesystem as their working environment.

The resource model is deliberately approximate. Commands use ordinary Rust data structures while
reserving modeled memory and charging stable abstract CPU units. This keeps the model predictable,
cheap, and easy to tune.

## Resource model

- **CPU** is monotonic fuel. Parsing, executor nodes, dispatch, input, output, and algorithms
  consume units. Exhaustion stops the evaluation.
- **Memory** is modeled concurrent working set. Command reservations are released on return;
  nested invocations contribute to the same peak.
- **Disk** is logical in-memory filesystem size. Content and a fixed 256-byte non-root node overhead count.
  Mutations that exceed quota roll back atomically, and deletion releases capacity.
- **Output** caps materialized stdout and stderr as a safety guardrail.

Defaults are 10,000,000 CPU units, 64 MiB memory, 64 MiB disk, and 4 MiB output. Costs are
deterministic rather than cycle-accurate. Results include a cost-model version.

## Build and use

```sh
cargo build --release

# Ordinary output
./target/release/shellsim -c 'printf "b\na\n" | sort'

# Host file used only as script source; execution occurs in a fresh simulated environment
./target/release/shellsim run script.sh arg1 arg2

# Persistent interactive session; state and resource usage accumulate until exit/exhaustion
./target/release/shellsim shell --cpu 100k --memory 8m --disk 2m --output 64k

# Structured evaluation report
./target/release/shellsim eval \
  --cpu 100k --memory 8m --disk 2m --output 64k \
  -c 'printf "b\na\n" | sort > result.txt; cat result.txt'
```

Limit values accept `k`, `m`, and `g` binary suffixes. Arguments after `--` in `eval` mode become
shell positional parameters.

The JSON report contains the exit status, typed stop reason, limits, aggregate usage, per-command
CPU/disk deltas, stdout, stderr, command trace, and unsupported capabilities.

## Persistent shell sessions

An `Environment` is a session, not a single command. Reusing it across `run_script_capture` calls
preserves the VFS, working directory, variables, arrays, functions, package state, clock/network
state, command history, and cumulative resource usage. CPU and output are cumulative fuel, disk
tracks current persistent usage, and temporary command memory is released while its peak remains.

`exit N`, `set -e` termination, CPU exhaustion, memory exhaustion, and output exhaustion make the
session terminal. Later calls return the same terminal outcome without executing or charging more
work. Disk-full errors are recoverable: a command can remove files and retry.

The `shell` subcommand drives one such environment line by line. It shows a prompt on a terminal,
preserves state between lines, exits normally on EOF or `exit`, and prints a reason before exiting
with status 137 when a resource is exhausted.

## Command implementations

Commands receive a uniform environment context:

```rust
fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32
```

The dispatcher applies each command's coarse base CPU and memory cost. Commands add dynamic costs
when useful:

```rust
if !env.reserve_memory(input.len() as u64 * 2) {
    return 137;
}
if !env.charge_cpu(input.len() as u64) {
    return 137;
}
```

New commands should live in their own module. Multiple names can share one behavioral module.
`echo`, `printf`, and `sort` demonstrate the layout. Older implementations still grouped by family
already use the same metered context and can be split mechanically when revised.

Disk enforcement lives inside `Vfs`, so direct command mutations cannot bypass capacity checks.
Commands should still surface `VfsError::NoSpace` with a non-zero status.

## Minimal Python compatibility

`python` is a bootstrap shim, not an embedded interpreter. It supports `--version`, a small
`python -c` subset for literal output, exit status, arguments and environment lookups, plus light
`pip`/`venv` compatibility. Unknown syntax fails loudly and is recorded as unsupported. Host
CPython is never invoked.

## Library API

```rust
use shellsim::{Environment, Limits};

let mut env = Environment::with_limits(Limits {
    cpu: 100_000,
    memory: 8 * 1024 * 1024,
    disk: 2 * 1024 * 1024,
    output: 64 * 1024,
});

let (outcome, stdout, stderr) = env.run_script_capture("echo hello");
```

`Interp` remains as an alias for `Environment` for source compatibility.

## Layout

```text
src/resources.rs       limits, accounting, outcomes, command usage
src/interp.rs          machine Environment and shell-local ProcessState
src/vfs.rs             quota-enforced in-memory filesystem
src/shell.rs           lexer, parser, capture API
src/expand.rs          shell expansion
src/exec.rs            metered executor, pipelines, redirects, control flow
src/commands/          registry, command context, implementations
src/python/mod.rs      minimal python -c shim
src/clock.rs           virtual clock
src/net.rs             virtual route-table network
```

Run the unit and resource-invariant tests with `cargo test`.
