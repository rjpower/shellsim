# Agent environment review and roadmap

Shellsim is currently a deterministic shell and Python workload simulator, not a general Unix
coding environment. Its strongest foundations are the capability boundary, virtual filesystem,
virtual clocks and network fixtures, resource accounting, and structured command telemetry. Its
largest compatibility gaps are shell correctness, processes, executable and build-tool behavior,
and a persistent agent-facing workspace protocol.

This document records the September 2026 review and the first implementation tranche. The initial
review findings are retained where they explain the roadmap; completion notes identify behavior
that is now implemented.

The first tranche added typed all-or-nothing shell parse failures, honest builtin failures, and
bounded VFS-only Git and shell-recipe Make subsets. The second added bounded logical process
records, child shell-state isolation, PID-backed jobs and `wait`, dynamic `ps`, and generated
read-only `/proc` plus finite `/dev` views. Concurrent job execution, descriptor routing, signals,
and the workspace protocol remain future work.

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
| Virtual filesystem | Useful core | First pseudo-files exist; permission enforcement and workspace diff remain |
| Virtual time | Strong | Signals and asynchronous Python waits remain incomplete |
| Virtual network | Useful fixture model | Request routes rather than sockets or running services |
| Shell grammar | Broad partial subset | Redirection, descriptor, and option behavior still has correctness gaps |
| Commands | Broad surface | Several partial operations or successful no-ops are misleading |
| Python | Substantial bounded interpreter | Arbitrary projects and third-party ecosystems remain out of scope |
| Processes and jobs | Cooperative logical processes | Background jobs, waits, sleeps, bounded pipelines, and default signal dispositions are modeled; process groups and handlers remain |
| Build ecosystem | Useful first slice | Git and Make are deliberately small; native compilation remains out of scope |
| Harness integration | Early foundation | No persistent machine protocol, workspace export, or trajectory runner |
| Observability | Good | Compatibility reporting is command-level rather than invocation-level |

## Observed shell gaps

The parser accepts functions, arrays, compound control flow, pipelines, command substitution,
heredocs, arithmetic, and common Bash options. The review nevertheless found behavior that an
agent cannot safely treat as Bash:

| Probe | Observed behavior | Required behavior |
|---|---|---|
| `X=outer; (X=inner); echo "$X"` | now prints `outer` | preserve child-state isolation |
| `printf x \| read X; echo "$X"` | pipeline state no longer leaks | preserve per-stage isolation |
| `sleep 1 & echo "$!"` | now returns a stable logical PID | add overlap only with a bounded scheduler |
| invoke a VFS executable through `PATH` | executable shell/Python scripts resolve in the VFS | extend formats only from observed needs |
| incomplete `if` statement | now rejected before execution | keep parser failure all-or-nothing and bounded |
| `cat /dev/null` | now succeeds with an empty read | add descriptor-backed devices separately |
| `git status` | modeled human and porcelain status | expand the coherent repository subset only as task evidence requires |
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
6. Resolve executable VFS scripts through `PATH` and honor executable metadata. Complete for
   shell/Python shebangs and explicit rejection of unsupported interpreters.
7. Add small common conveniences: aliases, directory stack, `mapfile`, jobs, wait, recursive
   search, patch application, and basic archive tools. Recursive search and atomic VFS-only patch
   application are now present.

Git and Make follow the same structure as the Python runtime: a small coherent state model
and parser, thin command-facing adapters, deterministic algorithms, explicit unsupported
frontiers, and tests against observable behavior. A simple slow implementation is preferable to
special cases in the shell executor. Git now has a VFS-native tree, blob, commit, and ref model
supporting the local edit/stage/commit/diff/history/branch/switch/restore/reset loop; remotes and
merge algorithms remain outside that baseline.

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

The process implementation creates logical children for isolation, `$$`, `BASHPID`, `$PPID`,
`$!`, dynamic `ps`, jobs, and wait status ownership. Background jobs, sleeps, waits, subshells, and
pipeline stages run through deterministic cooperative continuations. Pipeline descriptors use
bounded buffers with backpressure, so producers and consumers can overlap without eager whole-pipe
materialization. Command substitution and nested interpreter adapters remain synchronous and are
the next scheduler migration boundary.

Python `Popen`, `run`, `call`, `check_call`, and `check_output` execute only registered commands and
VFS shell or Python scripts. Live children use scheduler continuations and bounded descriptor
pipes; capture, inheritance, binary/text streams, duplex communication, cwd/env isolation,
status, signals, and virtual deadlines are modeled. The calling Python VM drives child quanta at
a nested cooperative boundary but is not itself resumable scheduler state. Arbitrary host
executables, `preexec_fn`, and native session manipulation remain rejected.

## Synthetic `/proc` and `/dev`

`/proc` is now a dynamic read-only view of modeled machine state rather than copied host data or
static VFS files. The first slice provides `/proc/self`, per-process `status`, `cmdline`, `environ`,
and `cwd`, plus deterministic `uptime`, `meminfo`, `cpuinfo`, `version`, and `mounts`.

The finite pseudo-device slice provides `/dev/null` and standard-descriptor links. Descriptor-backed
I/O is still required before those links have full read/write semantics. `/dev/zero` and random
devices require bounded streaming interfaces and are intentionally not exposed through eager file
reads. Signals can begin with `INT`, `TERM`, `KILL`, and `HUP` at scheduler boundaries.

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
5. Common agent conveniences. Bounded recursive `rg`, atomic patch application, and gzip streams
   are present; tar/zip containers and richer text tools remain.
6. Cooperative background scheduling, bounded pipes, default signal delivery, and live Python
   process handles are present; process groups, handlers, and scheduler-owned Python VM frames
   remain.
7. Persistent workspace protocol, evaluation scenarios, and external-agent experiments.

Every phase retains the existing security rule: simulated input can use only explicitly modeled
state and must never fall through to host filesystem, process, network, environment, or clock
capabilities.
