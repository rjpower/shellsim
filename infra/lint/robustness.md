# Robustness lane

Apply only the rules in this file. Findings must identify a concrete correctness, isolation, or
resource-accounting risk in changed code.

### `ml-env-var-vs-param` — internal behavior reads ambient configuration

Report a new environment-variable read below a process or CLI boundary when an explicit typed
parameter should carry the value. Boundary code that translates environment into configuration is
allowed. Simulated programs must never see the host environment implicitly.

### `ml-module-global-state` — mutable global state creates hidden coupling

Report new mutable process-global state, caches, or singletons whose lifetime can cross tests or
simulations. Immutable constants and explicitly synchronized registries with process-wide identity
are allowed.

### `ml-magic-constant` — an unexplained value controls behavior

Report repeated or non-obvious numeric/string literals that encode a protocol, resource limit, or
compatibility rule. Obvious local values and one-off test inputs are allowed.

### `ml-config-not-threaded` — configuration is accepted but bypassed

Report a new option that is parsed or stored but not delivered to every consumer, or a consumer
that still uses a hard-coded value after configuration was introduced.

### `ml-silent-fallback` — an error silently selects another behavior

Report `try`/`except`, `Result` recovery, defaults, or broad error mapping that masks a malformed
input or internal defect by taking a success path. Explicitly documented compatibility fallback
at an input boundary is allowed.

### `ml-error-swallow` — failure is discarded without preserving its meaning

Report ignored results, empty exception handlers, or error conversions that remove information
needed by the caller. Best-effort cleanup may suppress an error when the primary error is retained.

### `ml-guard-after-error` — validation occurs after an unsafe or expensive operation

Report checks for size, permissions, syntax, or capability that happen only after allocation,
host access, mutation, or other work the check was meant to prevent.

### `ml-ambient-host-capability` — simulated input can reach a host capability

Report any path by which simulated shell or Python input can access the host filesystem, process
table, network, environment, wall clock, randomness, or native execution. Explicit host-directory
mounts are allowed only through the existing scoped mapping boundary.

### `ml-unmetered-simulated-work` — input triggers work before resource charging

Report loops, allocation, output, filesystem growth, parsing, or expansion reachable from
simulated input that can grow without checked or saturating accounting before the host performs
the work.

### `ml-implicit-host-fallback` — unsupported simulation delegates to the host

Report fallback from an unsupported command, syntax form, module, or device to host execution or
host I/O. Unsupported behavior must fail explicitly inside the simulator.
