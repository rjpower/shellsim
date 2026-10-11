# Cruft and tests lane

Apply only the rules in this file. Prefer deletion or an existing seam over parallel machinery,
but respect compatibility contracts and explicit security boundaries.

Use [TESTING.md](../../TESTING.md) when judging tests. Each test must justify its maintenance cost
with a distinct bug it catches. Prefer regressions; unit tests need behavior that is difficult to
exercise or diagnose through a regression. Do not request more cases merely to increase coverage.

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
cases would preserve meaningful variation. Recommend deleting equivalent cases before suggesting
parametrization. Keep separate tests when setup or failure meaning differs.

### `ml-redundant-test-coverage` — added coverage repeats an existing behavioral claim

Report added cases, tables, or test layers that exercise the same failure mode as existing
coverage. Name the existing test and the repeated claim. Also report unit tests of behavior
already exercised by a regression when they add no distinct invariant or failure signal. Do not
infer redundancy from test size or similar names. Separate isolation boundaries and independent
compatibility oracles can justify multiple tests.

### `ml-slop-test` — assertions do not prove the stated behavior

Report tests that only check "no error," use tautological assertions, swallow failures, or accept
many unrelated outputs. Also report assertions that merely repeat static type guarantees,
constructor assignments, configuration projections, constant values, internal helper dispatch,
or incidental wording. A configuration projection only checks that a copied or lowered field
equals the value supplied by the test; an external wire translation can be a real contract.
Standalone scalar/configuration guard tests need a reported regression, public compatibility
contract, or simulation safety consequence. Identify the incidental fact being pinned and ask
for deletion or a behavioral regression, not more assertions on the same implementation detail.

Exact fixtures are appropriate when shell/Python compatibility or a machine-readable output
format is the contract. Guest-visible Python types and exception types are compatibility behavior;
they are not redundant host type checks. Focused private-invariant tests are allowed when a
regression would be difficult to construct or diagnose.

Successful completion alone can be a valid regression for a panic on a specific input. Otherwise,
`is_ok()` or `is_err()` must not substitute for the claimed output, error variant, or state change.
Report permanently ignored tests and empty stubs; temporary exclusions with a concrete reason
and restoration condition are allowed.

### `ml-time-sleep-in-test` — elapsed host time controls a test

Report sleeps, retries driven by wall time, or timing thresholds. Shellsim tests must use virtual
time or explicit state transitions.

### `ml-unsupported-frontier-uncovered` — a new capability lacks boundary tests

Report a new or changed capability boundary with no regression for a concrete isolation,
resource-exhaustion, malformed-input, or unsupported failure the change can introduce. Name the
failure and check existing coverage first. Do not demand one test per guard or category.
