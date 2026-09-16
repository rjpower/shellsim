# Cruft and tests lane

Apply only the rules in this file. Prefer deletion or an existing seam over parallel machinery,
but respect compatibility contracts and explicit security boundaries.

### `ml-rollout-scaffolding` — temporary migration machinery lacks a removal trigger

Report dual paths, temporary aliases, or rollout flags introduced without a concrete compatibility
need and durable removal condition. Published API compatibility can justify a shim.

### `ml-obsolete-after-refactor` — changed code leaves its replaced path behind

Report an old helper, branch, field, or test made unreachable by the same change. Confirm that
registries, command dispatch, and string lookup do not still reference it.

### `ml-add-then-remove` — the change performs work it immediately undoes

Report values inserted then removed, data converted through an avoidable intermediate shape, or
state toggled back within one path when no observable boundary requires the intermediate state.

### `ml-speculative-abstraction` — abstraction has no present responsibility

Report a wrapper, trait, generic parameter, or factory that serves one trivial use and does not
clarify behavior. An explicit capability, security, correctness, serialization, or test boundary
is sufficient present value even before reuse.

### `ml-duplicate-logic-block` — a third copy repeats the same algorithm

Report a third structurally equivalent implementation introduced in one layer. Two copies may
remain until the pattern is established. Earlier extraction is warranted when it centralizes a
security, compatibility, or correctness invariant.

### `ml-parallel-source-impl` — two production paths implement the same contract

Report a new side-by-side implementation selected by a flag or suffix when one strategy boundary
or direct migration would suffice. Independent backends with meaningfully different contracts are
allowed.

### `ml-test-double-mirrors-prod` — test code reimplements the production algorithm

Report a fake or expected-value helper that duplicates the implementation under test and can pass
with the same bug. Small independent reference models and external compatibility oracles are
allowed.

### `ml-duplicate-test-body` — tests copy setup and assertions without a semantic distinction

Report repeated test bodies that differ only in inputs and expected outputs when a named table of
cases would preserve intent. Keep separate tests when setup or failure meaning differs.

### `ml-slop-test` — assertions do not prove the stated behavior

Report tests that only check "no error," use tautological assertions, swallow failures, or accept
many unrelated outputs. Exact fixtures are appropriate when shell/Python compatibility or a
machine-readable output format is the contract.

### `ml-time-sleep-in-test` — elapsed host time controls a test

Report sleeps, retries driven by wall time, or timing thresholds. Shellsim tests must use virtual
time or explicit state transitions.

### `ml-unsupported-frontier-uncovered` — a new capability lacks boundary tests

Report a new parser, VM, command, device, or stdlib feature whose tests cover success but omit at
least one directly adjacent malformed, resource-exhaustion, or explicitly unsupported case.
