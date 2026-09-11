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

For development, `make format`, `make lint`, and `make test` are the canonical local commands and
the exact entrypoints used by CI. See [CONTRIBUTING.md](CONTRIBUTING.md) for code, testing, review,
and optional pre-commit-hook guidelines.

The JSON report contains the exit status, typed stop reason, limits, aggregate usage, per-command
CPU/disk deltas, stdout, stderr, command trace, and unsupported capabilities.

## Virtual time

An environment owns one deterministic `Timeline`; neither shell commands nor the Python engine
read the host clock or block a host thread. Its clock domains are intentionally separate:

- **Monotonic time** starts at zero and orders sleeps, deadlines, and injected events.
- **Wall time** is the fixed default epoch `2025-01-01T00:00:00Z` plus monotonic time and an
  explicit adjustment. Adjusting wall time never changes a deadline.
- **Process CPU time** derives from deterministic resource fuel (one CPU unit is one virtual
  microsecond) and never advances monotonic or wall time.

Scheduled events use `(deadline_ns, insertion_sequence)` ordering, so simultaneous events replay
in a stable order. Pending-event count and scheduling horizon are bounded. Cloning a `Timeline`
captures its clocks, ordering sequence, pending events, and ready events; no hidden host state is
needed to replay it.

`sleep` and Python `time.sleep()` schedule a wake event. When the current executor has no runnable
work, it jumps directly to the next event. `timeout` schedules a deadline event and interrupts
nested shell or Python execution at that instant. `date`, Python `time.time*`, filesystem mutation
times, and those deadlines all observe the same environment timeline. Python `time.monotonic*`,
`perf_counter*`, and `process_time*` expose their corresponding domains.

Runnable commands and bytecode consume zero virtual duration unless an operation explicitly
models latency. Thus a timeout is observed at a blocking/yield point, while CPU fuel bounds a
zero-time busy loop; shellsim does not invent a host-dependent instructions-per-second rate.

The executor still evaluates `&` jobs synchronously, so independent background sleeps do not yet
overlap. The timeline/event contract is designed for the next scheduler step: background AST
frames become resumable tasks, runnable tasks execute in stable task-id order, and time advances
only when the runnable set is empty. This limitation is explicit rather than approximating
concurrency by guessing durations.

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

Invoking `python` without arguments transfers the foreground session to a deliberately-minimal
Python REPL. Simple assignments and expressions persist across actions; `exit()` or `quit()`
returns to the shell. This is a modeled process mode, not access to host CPython.

## Bash-ish compatibility

The shell intentionally targets common agent-written Bash rather than the full Bash grammar. It
supports functions, indexed and associative arrays, `if`/`case`/`for`/`while`/`until`, C-style
`for ((...))` loops, `((...))`, pipelines, `&&`/`||`, background jobs, groups and subshells,
heredocs and here-strings, command/arithmetic substitution, brace expansion, parameter expansion,
globbing, `[[...]]`, and frequently used `set` options including `pipefail`.

Standard paths such as `/bin/sh` and `/usr/bin/env` resolve to their simulated commands. More
specialized Bash behavior—process substitution, traps/signals, coprocesses, job timing, and exact
subshell isolation—remains outside the faithful subset.

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

The current command set includes filesystem and text coreutils, `grep`, `sed`, a useful partial
`awk`, hashes and encoders, virtual `curl`/`wget`, shell builtins, minimal package/Python launchers,
and simulated system queries such as `env`, `printenv`, `uname`, `id`, `nproc`, `df`, `free`, and
`ps`. Partial commands are surfaced in evaluation reports instead of being presented as fully
faithful implementations.

Disk enforcement lives inside `Vfs`, so direct command mutations cannot bypass capacity checks.
Commands should still surface `VfsError::NoSpace` with a non-zero status.

## Python 3.14 compatibility

`python`, `python3`, and `python3.14` route to shellsim's safe in-process interpreter. Source goes
through a UTF-8/indentation-aware lexer, owned AST, semantic bytecode compiler, and metered stack
VM; host CPython is never invoked. The current language slice covers scalar and mutable containers,
comparisons and control flow, functions/closures/defaults/`*args`, classes and bound methods,
user inheritance with C3 lookup, `int` subclasses, constrained metaclasses, comprehensions,
suspended generators, exceptions and context managers, `assert`, decorators,
starred assignment/calls, f-strings, VFS-only imports, common iterator/container builtins, and the
modeled REPL/script/stdin/shebang entrypoints. Unsupported syntax and APIs fail loudly with a
diagnostic.

Native Python modules use the erased value ABI and checked object views described in
[`PYTHON_RUNTIME_MODEL.md`](PYTHON_RUNTIME_MODEL.md). The layout keeps module definitions small
while routing allocation, recursion, and work through the interpreter's resource meter.

The requested stdlib gate is 18/18 exact CPython 3.14 probes for these APIs: `sys.executable`,
`os.getenv`, `collections.defaultdict`, `itertools.count`/`islice`, `heapq.heapify`/`heappop`,
`bisect.bisect_left`, `math.sqrt`/`ceil`, `string.digits`, `json.dumps(sort_keys=...)`, `re.sub`,
`functools.reduce`, `dataclasses.dataclass`, `typing.List[...]`, `enum.Enum`,
`argparse.ArgumentParser.prog`, `import subprocess`, and the `pytest`/`unittest.TestCase` entry
points. These are intentionally partial module slices, not claims of complete stdlib support.

`pytest` and `unittest` are VFS-only first runner slices: explicit files, stable definition-order
collection, plain zero-argument pytest tests, direct `unittest.TestCase` classes, tested assertions/
skip/raises controls, and bounded wrappers. Fixtures, decorated tests, plugins, rich
parametrization, async fixtures, directory/package discovery, and unlisted flags are rejected
explicitly. The 100-row TaskTrove mini corpus is differential-tested with per-row provenance (99
supported, one async frontier), and one complete `build-system-task-ordering` solution matches
CPython 3.14. CPU fuel, modeled memory, output, source/wrapper size, and nesting limits keep this
general-purpose slice safe and deliberately slow.

See [`PYTHON_3_14_PROPOSAL.md`](PYTHON_3_14_PROPOSAL.md) for the architecture, target extensions,
module matrix, and validation plan.

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
src/commands/awk.rs    partial record-oriented awk
src/commands/system.rs simulated environment/system queries
src/python/            Python 3.14 lexer, parser, bytecode compiler, and metered VM
src/clock.rs           virtual clock
src/net.rs             virtual route-table network
```

Run the unit and resource-invariant tests with `cargo test`.
