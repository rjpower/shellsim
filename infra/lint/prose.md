# Naming and documentation lane

Apply only the rules in this file. Documentation means Rust doc comments, Python docstrings, and
user-facing prose as appropriate. Do not demand documentation that merely repeats clear code.

### `ml-misleading-name` — a name contradicts behavior

Report a symbol or option whose name asserts the wrong unit, side effect, scope, or outcome.

### `ml-vestigial-qualifier` — a legacy or version suffix has lost its contrast

Report names such as `new`, `old`, `legacy`, or `v2` when the counterpart no longer exists or the
qualifier conveys no stable contract.

### `ml-abbreviated-name` — a nonstandard abbreviation obscures meaning

Report public or long-lived names whose abbreviation is not conventional in the surrounding code.
Short loop indices and domain-standard terms are allowed.

### `ml-restating-comment` — prose translates the next line without adding intent

Report comments or docs that only narrate mechanics. Preserve comments that explain an invariant,
compatibility constraint, capability boundary, or non-obvious reason.

### `ml-implementation-doc` — public documentation promises an internal mechanism

Report API documentation centered on private steps rather than observable behavior, invariants,
inputs, errors, and resource effects. Mechanism belongs next to code only when a maintainer could
otherwise break a required invariant.

### `ml-pr-reference-comment` — source prose relies on transient review context

Report comments that name a pull request, sprint, task phase, or temporary branch. Stable issue
URLs and architecture records are allowed when they remain necessary to understand the code.

### `ml-bare-todo` — a TODO has no durable trigger

Report TODOs without a specific condition or issue that makes the next action discoverable. A
named human owner is not required.

### `ml-stale-documentation` — documentation describes previous behavior

Report comments, docs, examples, or option descriptions contradicted by the changed implementation.

### `ml-undocumented-outcome` — a non-obvious result or error has no contract

Report public behavior whose result variants, resource-exhaustion response, or unsupported case
cannot be inferred from its name and lacks documentation.

### `ml-rotting-historical-reference` — history is recorded instead of current intent

Report prose that explains how a migration unfolded rather than the invariant that remains. Keep
history in the pull request or commit; keep the present constraint in source.
