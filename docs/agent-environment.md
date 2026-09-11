# Agent environment review and roadmap

Shellsim is currently a deterministic shell and Python workload simulator, not a general Unix
coding environment. Its strongest foundations are the capability boundary, virtual filesystem,
virtual clocks and network fixtures, resource accounting, and structured command telemetry. Its
largest compatibility gaps are shell correctness, processes, executable and build-tool behavior,
and a persistent agent-facing workspace protocol.

This document records the September 2026 review and the first implementation tranche. The initial
review findings are retained where they explain the roadmap; completion notes identify behavior
that is now implemented.

The first tranche added typed all-or-nothing shell parse failures with a parser progress invariant,
made unsupported no-op builtins fail explicitly, exposed already-completed background jobs through
`jobs` and `wait`, and added bounded VFS-only Git and shell-recipe Make subsets. Process isolation,
concurrent job execution, pseudo-filesystems, and the workspace protocol remain future work.

## Product boundary

The achievable near-term target is a useful Unix-shaped environment for shell- and Python-centric
agent evaluation. Native programs, arbitrary package installation, real sockets, and complete
Linux syscall behavior remain outside the simulation. A conventional container or microVM is a
more honest backend when a task requires those capabilities.

The actual Codex or Claude client should run outside shellsim. Model access needs credentials,
networking, and a native client runtime. A later host adapter can expose a persistent shellsim
`Environment` as shell, file, patch, and inspection tools without granting simulated code host
capabilities.

## Capability assessment

| Dimension | Current assessment | Principal limitation |
|---|---|---|
| Isolation and determinism | Strong | Seccomp is defense in depth, not complete host filesystem confinement |
| Resource limits | Strong | Some expansion paths still need tighter preallocation bounds |
| Virtual filesystem | Useful core | No pseudo-filesystems, devices, permission enforcement, or workspace diff |
| Virtual time | Strong | Background sleeps do not overlap |
| Virtual network | Useful fixture model | Request routes rather than sockets or running services |
| Shell grammar | Broad partial subset | Isolation, redirection, and options still have correctness gaps |
| Commands | Broad surface | Several partial operations or successful no-ops are misleading |
| Python | Substantial bounded interpreter | Arbitrary projects and third-party ecosystems remain out of scope |
| Processes and jobs | Minimal | One mutable process, synchronous jobs, and static process reporting |
| Build ecosystem | Useful first slice | Git and Make are deliberately small; native compilation remains out of scope |
| Harness integration | Early foundation | No persistent machine protocol, workspace export, or trajectory runner |
| Observability | Good | Compatibility reporting is command-level rather than invocation-level |

## Observed shell gaps

The parser accepts functions, arrays, compound control flow, pipelines, command substitution,
heredocs, arithmetic, and common Bash options. The review nevertheless found behavior that an
agent cannot safely treat as Bash:

| Probe | Observed behavior | Required behavior |
|---|---|---|
| `X=outer; (X=inner); echo "$X"` | prints `inner` | prints `outer` |
| `printf x \| read X; echo "$X"` | prints `x` | child pipeline state does not leak |
| `sleep 1 & echo "$!"` | `$!` is not a PID | stable child PID |
| invoke a VFS executable through `PATH` | command not found | resolve and execute it |
| incomplete `if` statement | now rejected before execution | keep parser failure all-or-nothing and bounded |
| `cat /dev/null` | missing file | successful empty read |
| `git status` | modeled short/porcelain status | expand the coherent repository subset only as task evidence requires |
| `make test` | executes explicit shell recipes | add Make syntax deliberately and reject unsupported constructs |

Several formerly silent builtins, including process controls, aliases, directory-stack commands,
and `mapfile`, now fail explicitly. Unsupported behavior should remain observable; success must
mean that the requested effect occurred.

### Shell priorities

1. Preserve typed parse errors and reject incomplete or unsupported syntax before execution.
2. Keep command trust classifications honest; do not introduce silent successful no-ops.
3. Isolate subshell, command-substitution, and pipeline shell state while sharing machine state.
4. Make redirection failures, descriptor duplication, and pseudo-device behavior correct.
5. Enforce declared shell options such as `nounset` and `xtrace`.
6. Resolve executable VFS scripts through `PATH` and honor executable metadata.
7. Add small common conveniences: aliases, directory stack, `mapfile`, jobs, wait, recursive
   search, patch application, and basic archive tools.

Git and Make should follow the same structure as the Python runtime: a small coherent state model
and parser, thin command-facing adapters, deterministic algorithms, explicit unsupported
frontiers, and tests against observable behavior. A simple slow implementation is preferable to
special cases in the shell executor.

## Minimal process model

Host processes and preemptive host threads are not required. A deterministic logical model can
begin synchronously:

```text
Machine
  VFS
  Timeline
  VirtualNet
  Resources
  ProcessTable
  next_pid

Process
  pid, ppid, process_group
  argv, cwd, environment
  shell state
  descriptor table
  state: Runnable | Sleeping | Exited(status)
```

Creating logical child records immediately provides correct state isolation, `$$`, `$PPID`, `$!`,
dynamic `ps`, jobs, wait, and an ownership model for exit status. Pipelines may initially retain
materialized bounded buffers while running each stage in a child snapshot. Cooperative scheduling
can follow at explicit blocking points such as sleep, pipe I/O, process wait, and virtual service
operations.

Python `subprocess.run` and `check_output` can eventually execute only registered commands and VFS
shell or Python scripts. Arbitrary host executables, `preexec_fn`, and native session manipulation
must remain rejected.

## Synthetic `/proc` and `/dev`

`/proc` should be a dynamic read-only view of modeled machine state rather than copied host data or
static VFS files. A useful first slice is `/proc/self`, per-process `status`, `cmdline`, `environ`,
and `cwd`, plus deterministic `uptime`, `meminfo`, `cpuinfo`, `version`, and `mounts`.

Typed pseudo-devices should cover `/dev/null`, `/dev/zero`, `/dev/stdin`, `/dev/stdout`, and
`/dev/stderr`. A seeded deterministic `/dev/urandom` may be added if task evidence requires it.
Signals can begin with `INT`, `TERM`, `KILL`, and `HUP`, delivered only at scheduler boundaries.

These views must never proxy the host's `/proc`, devices, processes, or random source.

## Later agent workspace mode

After shell, Git, Make, and process behavior are useful, add a host-side persistent protocol around
`Environment`. A small NDJSON service is sufficient:

- create an episode from a bounded directory snapshot;
- execute one shell action with explicit stdin;
- read, write, list, stat, and apply a patch in the VFS;
- return a canonical diff against the initial snapshot;
- report resource use, commands, unsupported operations, network requests, and processes;
- reset or clone a deterministic snapshot when branching evaluation needs it.

Binary fields should use an explicit byte encoding rather than lossy JSON strings. The existing
Python project ingestion should be factored into a generic trusted harness boundary. A thin MCP or
model-API adapter can then connect an external agent to the service. Running the agent executable
inside the simulation is not a goal.

An evaluation scenario should define initial files, environment, limits, virtual time and network
fixtures, the action or agent adapter, and semantic assertions over status, files, unsupported
operations, and resources. Its result should include a replayable action transcript and canonical
workspace patch. Strict evaluation should fail when a no-op or unsupported feature is used.

## Ordered roadmap

1. Shell honesty and correctness: parse errors, truthful trust levels, isolation, redirection,
   options, and `PATH`.
2. Clean Git and Make foundations with useful deterministic subsets.
3. Logical process records, jobs, wait, descriptors, and dynamic `ps`.
4. Synthetic `/proc` and `/dev`, then bounded Python subprocess operations.
5. Common agent conveniences such as recursive search, patching, archives, and richer text tools.
6. Cooperative background scheduling, bounded pipes, and a small signal model.
7. Persistent workspace protocol, evaluation scenarios, and external-agent experiments.

Every phase retains the existing security rule: simulated input can use only explicitly modeled
state and must never fall through to host filesystem, process, network, environment, or clock
capabilities.
