# shellsim

Shellsim is a BusyBox for containers: one small, deterministic process that provides a useful
Unix-shaped environment without starting a VM, container runtime, or host subprocess. It is built
for experimentation and testing with reinforcement-learning rollouts and agentic environments,
where fast startup, reproducibility, isolation, and explicit resource limits matter more than
cycle-accurate emulation.

Shell programs, common command-line tools, logical processes, and Python run in-process against an
in-memory filesystem. Simulated code cannot access the host filesystem, processes, network,
environment, or clock. A trusted harness may copy a selected project into the virtual filesystem
before execution; changes never write back to the host.

## Compatibility

Shellsim aims for broad compatibility inside clear boundaries. A supported facility should handle
almost all ordinary uses, even when obscure flags or legacy behavior remain out of scope. A module
or command with no coherent useful subset is omitted instead of being exposed as a misleading
stub. Unsupported syntax, options, executable formats, and capabilities fail visibly and are
included in structured results.

The current environment includes:

- a Bash-like shell with pipelines, redirections, functions, common expansions, control flow,
  background jobs, signals, and job control;
- common filesystem, text, archive, Git, Make, process, and system commands;
- deterministic virtual time, network fixtures, `/proc`, `/dev`, processes, descriptors, and
  bounded pipes;
- a mostly complete Python language runtime with a deliberately selected standard-library and
  third-party module surface.

Python is source-compatible where supported, not ABI-compatible with CPython. Native extensions,
package installation, compilers, and arbitrary machine code are outside the simulation boundary.
See [Python in shellsim](docs/python.md) for the current contract.

## Install and run

Install the Python package and console command:

```sh
python -m pip install shellsim
shellsim -c 'printf "b\na\n" | sort'
shellsim --root ./project -c 'python3.14 test.py'
```

`--root` copies the selected host tree into a disposable `/work` snapshot. With no `-c` and a
terminal attached, `shellsim` starts a persistent interactive session.

To build the Rust binaries from source:

```sh
cargo build --release
./target/release/shellsim -c 'echo hello'
./target/release/shellsim eval --cpu 100k --memory 8m -c 'make test'
./target/release/shellsim-python ./project/main.py -- arg1
```

Limits accept `k`, `m`, and `g` binary suffixes. `eval` emits a structured result containing the
exit status, stdout and stderr, resource use, command trace, and unsupported behavior.

The Python API exposes fresh and persistent environments:

```python
import shellsim

environment = shellsim.Environment(cpu=100_000)
environment.write_file("/work/main.py", "print(6 * 7)\n")
result = environment.run("python3.14 /work/main.py")
assert result.returncode == 0
assert result.stdout == b"42\n"
```

## Agent harness

`shellsim serve --root ./project` runs a persistent newline-delimited JSON session. It supports
bounded execution, streaming actions, VFS operations, checkpoints, workspace diffs, process and
resource inspection, and deterministic session forks. `shellsim mcp` exposes the same environment
as a stdio MCP server. `shellsim replay scenario.ndjson` reruns checked action transcripts.

```sh
printf '%s\n' \
  '{"id":1,"op":"execute","source":"printf hello > result"}' \
  '{"id":2,"op":"workspace_diff"}' \
  | shellsim serve --root ./project
```

## Resource model

CPU is deterministic fuel, memory is modeled working set, disk is current virtual-filesystem
usage, and output bounds materialized stdout and stderr. The defaults are 10,000,000 CPU units,
64 MiB memory, 64 MiB disk, and 4 MiB output. Costs are stable and intentionally approximate.
Exhaustion is observable and never falls back to an ambient host implementation.

For internals and contribution workflow, see [implementation](docs/implementation.md),
[Python](docs/python.md), and [CONTRIBUTING.md](CONTRIBUTING.md).
