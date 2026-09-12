# shellsim

`shellsim` is a deterministic, resource-constrained BusyBox-like environment for evaluating
agents. Shell programs and Unix-style commands run in-process against an in-memory filesystem;
they never execute host programs or use the host filesystem as their working environment.

The resource model is deliberately approximate. Commands use ordinary Rust data structures while
reserving modeled memory and charging stable abstract CPU units. This keeps the model predictable,
cheap, and easy to tune.

See [docs/agent-environment.md](docs/agent-environment.md) for the reviewed gap between the current
simulator and a useful Unix-shaped coding-agent harness, plus the ordered implementation roadmap.

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

# Import a host Python project into a fresh VFS and run it in shellsim
./target/release/shellsim-python project/main.py -- arg1
./target/release/shellsim-python project/tests --pytest
./target/release/shellsim-python --json --root project project/main.py
```

Limit values accept `k`, `m`, and `g` binary suffixes. Arguments after `--` in `eval` mode become
shell positional parameters.

For development, `make format`, `make lint`, and `make test` are the canonical local commands and
the exact entrypoints used by CI. See [CONTRIBUTING.md](CONTRIBUTING.md) for code, testing, review,
and optional pre-commit-hook guidelines.

The JSON report contains the exit status, typed stop reason, limits, aggregate usage, per-command
CPU/disk deltas, stdout, stderr, command trace, and unsupported capabilities.

`shellsim-python` treats its host path as trusted harness input, copies the containing project into
`/work`, then closes that boundary before simulated Python starts. A directory automatically
discovers `test_*.py` files; `--entry FILE` selects a script within a directory. Use `--root` to
control which project tree is imported and the standard limit flags to constrain the run.

## Virtual time

An environment owns deterministic monotonic, wall, and process-CPU clocks. Sleeps and deadlines
advance the event queue without blocking a host thread; VFS timestamps and Python observe the same
timeline. Runnable work has zero virtual duration and is bounded by CPU fuel. Background jobs still
execute synchronously, so independent sleeps do not overlap. See
[docs/implementation.md](docs/implementation.md) for the state, scheduler, and replay contracts.

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
with status 137 when a resource is exhausted. It is an action console rather than a resumable
terminal: each completed action has closed stdin. Use a pipe or heredoc for command input. The
console collects a heredoc through its terminating delimiter before executing the action.

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
specialized Bash behavior, including process substitution, custom traps, coprocesses, process
groups, and some descriptor forms, remains outside the faithful subset. Logical children provide
isolated shell state, stable PIDs, overlapping virtual-time jobs, bounded pipes, `jobs`/`wait`,
default signal delivery, dynamic `ps`, and generated `/proc` views without creating host processes.

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

New commands should live in a focused module and use only the modeled command context. See
[docs/implementation.md](docs/implementation.md) for the integration checklist, trust levels,
resource rules, and the reason native compilers remain outside the simulation.

The current command set includes filesystem and text coreutils, `grep`, `sed`, a useful partial
`awk`, hashes and encoders, bounded tar and gzip tools, virtual `curl`/`wget`, deterministic Git and
Make subsets, shell builtins, minimal package/Python launchers, and simulated system queries such
as `env`, `printenv`, `uname`, `id`, `nproc`, `df`, `free`, and `ps`. Partial commands are surfaced
in evaluation reports instead of being presented as fully faithful implementations.

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

Native Python modules use an erased value ABI, checked object views, declarative type/module tables,
and narrow modeled capabilities. See [docs/python.md](docs/python.md) for the goals, value and
object model, extension workflow, compatibility evidence, and explicit frontiers.

The requested stdlib gate is 21/21 exact CPython 3.14 probes for these APIs: `sys.executable`,
`os.getenv`, `collections.defaultdict`, `itertools.count`/`islice`, `heapq.heapify`/`heappop`,
`bisect.bisect_left`, `math.sqrt`/`ceil`, `string.digits`, `json.dumps(sort_keys=...)`, `re.sub`,
`functools.reduce`, `dataclasses.dataclass`, `typing.List[...]`, `enum.Enum`,
`argparse.ArgumentParser.prog`, `csv.reader`/`writer`, source-backed `Counter`, `deque`, `json`,
`os.path`, `datetime`, byte-preserving `base64`, `hashlib`, `struct`, and `zlib`,
`import subprocess`, and the
`pytest`/`unittest.TestCase` entry points. These are intentionally partial module slices, not
claims of complete stdlib support.

`pytest` and `unittest` are VFS-only first runner slices: explicit files, stable definition-order
collection, plain zero-argument pytest tests, direct `unittest.TestCase` classes, tested assertions/
skip/raises controls, and bounded wrappers. Fixtures, decorated tests, plugins, rich
parametrization, async fixtures, directory/package discovery, and unlisted flags are rejected
explicitly. The 100-row TaskTrove mini corpus is differential-tested with per-row provenance (99
supported, one async frontier), and one complete `build-system-task-ordering` solution matches
CPython 3.14. CPU fuel, modeled memory, output, source/wrapper size, and nesting limits keep this
general-purpose slice safe and deliberately slow.

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

// Harness actions may attach stdin without giving the simulated command host-terminal access.
let (outcome, stdout, stderr) =
    env.run_script_capture_with_stdin("cat > input.txt", b"hello\n");
```

An `Environment` preserves its VFS, working directory, variables, functions, options, and resource
usage across actions. Each action receives its own explicit stdin byte stream; an input redirect in
the action takes precedence. New environments include `/root`, `/tmp`, and `/work`.

`Interp` remains as an alias for `Environment` for source compatibility.

## Layout

```text
src/resources.rs       limits, accounting, outcomes, command usage
src/interp.rs          machine Environment and shell-local ProcessState
src/process.rs         bounded logical process identities and lifecycle
src/pseudo_fs.rs       generated read-only /proc and finite /dev views
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
docs/implementation.md architecture and command integration guide
docs/python.md         Python goals, runtime model, and extension guide
docs/agent-environment.md reviewed agent-harness gaps and roadmap
```

Run the unit and resource-invariant tests with `cargo test`.
