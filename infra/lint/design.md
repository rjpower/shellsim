# Design lane

Apply only the rules in this file. Judge responsibilities and API shape from behavior, not raw
line counts.

For commands, workers, and orchestration changes, trace the request through planning, execution,
storage, and export. Inspect existing entry points and adapters, including unchanged code, before
deciding whether a new path is necessary. A small wrapper can introduce a second workflow.

### `ml-work-outside-graph` — orchestration hides a declared workflow dependency

In an existing graph-based workflow, report a new prerequisite, producer, packaging step, or check
that is invoked out of band or required as manual setup when the graph can represent it. Name the
producer, consumer, and missing edge, and explain the resulting ordering, caching, or planning
gap. Process startup, authentication, and worker allocation may remain backend concerns.

### `ml-duplicate-workflow` — a new entry point repeats workflow semantics

Report a command or worker that adds another planner, graph walker, cache policy, or result
publication path for the same contract. Identify the existing execution seam and the duplicated
decision. Local invocation and remote dispatch should share worker logic; transport, allocation,
and genuinely different adapter contracts do not need one implementation. Do not request a
generic framework or combine unrelated operations merely because they run on workers.

### `ml-storage-placement-coupling` — execution placement selects storage semantics

Report orchestration that ties local or remote execution to an artifact store, or branches on
storage schemes where an existing storage API handles the operation. Name that API and the
affected supported combination. Compiler workspaces and materialized inputs may use local paths;
worker access constraints may be checked explicitly at the boundary.

### `ml-cache-hit-does-work` — reusable output still triggers avoidable production work

Trace fresh and cached requests. Report newly introduced prerequisite materialization, rebuilds,
artifact scans, transfers, or packaging before a usable cache record is checked. Also report
placement or unrelated inputs added to a product key without affecting its output. Name the
record or declared input that can decide reuse and the concrete unnecessary work. Runtime
dependencies and checks explicitly requested by the caller still need to run or be available;
required integrity checks are allowed. Do not propose TTLs or broader fingerprinting by default.

### `ml-overloaded-function` — one function owns unrelated jobs

Report a function that interleaves at least three separately nameable responsibilities with no
useful internal boundary. A long, linear implementation of one operation is allowed.

### `ml-monolithic-function` — flags create several operating modes

Report a function whose boolean or mode knobs produce multiple independent behaviors that should
be separate entry points or typed strategies. Simple two-state parser or validation strictness is
allowed.

### `ml-god-module` — a type or module mixes unrelated responsibility clusters

Report a type or module whose members form distinct clusters that do not share state or an
invariant. Large but cohesive interpreters and state machines are allowed.

### `ml-reverse-layer-dependency` — dependency points against the architecture

The Rust core must not depend on the binary, PyO3 adapter, or Python API. The binary and adapter
may depend on the core; the Python API may depend on the native adapter. Report a new dependency
that reverses this direction or makes the core aware of a presentation layer.

### `ml-bool-flag-arg` — a boolean hides a behavioral choice

Report a boolean argument when call sites cannot communicate the selected behavior clearly or a
third state is plausible. Obvious predicates and stable binary toggles are allowed.

### `ml-bool-return-status` — a boolean collapses distinct outcomes

Report a boolean result when callers need to distinguish more than two outcomes, such as success,
timeout, resource exhaustion, and unsupported behavior. Genuine predicates are allowed.

### `ml-tuple-return-shape` — positional output hides field meaning

Report a fixed tuple of three or more semantically distinct values that should be a named struct,
dataclass, or typed record. Coordinates, key/value pairs, and homogeneous sequences are allowed.

### `ml-missing-protocol` — callers depend on an implementation instead of a boundary

Report repeated ad hoc duck typing, callbacks, or concrete-type coupling where a small Rust trait
or Python protocol would make an existing boundary explicit. Do not request an interface for one
call site or merely to anticipate reuse.

### `ml-untyped-record` — a structured value is passed as a loose map

Report a map with a stable, known schema that crosses module boundaries or is unpacked repeatedly.
Dynamic JSON, environment maps, and command dictionaries with genuinely open keys are allowed.
