# Testing policy

Tests are a maintenance cost. Keep a test only when the bug it would catch justifies the code,
fixtures, runtime, and constraints it adds to future changes. Test count and coverage percentage
are not goals. Prefer regression tests over unit coverage.

This policy adapts [Marin's testing policy](https://github.com/marin-community/marin/blob/main/TESTING.md)
and [marin-style's testing guidelines](https://github.com/marin-community/marin-style/blob/d855d6304ad0532c131ef2e208db84ed7acf4721/src/marin_style/assets/agents/TESTING-core.md)
to Rust and shellsim's simulation boundary. Read it and the nearest `AGENTS.md` before writing or
reviewing tests. See [CONTRIBUTING.md](CONTRIBUTING.md#testing) for commands and test locations.

A test must fail when the behavior it claims to protect is wrong. It should survive a harmless
helper rename, internal refactor, or log-message edit. Omitting a low-value test is better than
checking it in merely to show that testing happened.

## Choose the smallest useful regression

- For a bug fix, reproduce the failure through an existing public API, shell command, or guest
  program and assert the corrected result. Prefer a test that fails before the fix.
- For new behavior, exercise the observable contract with a representative scenario. Assert
  output, simulated filesystem state, status, or another effect a caller relies on.
- Search existing coverage first. Extend a regression when it can express the new failure
  clearly. Add a separate case only for a distinct failure mode.
- Use unit tests for behavior that would be difficult to exercise or diagnose through a
  regression: for example, arithmetic overflow at an internal accounting boundary that would
  otherwise require an enormous guest workload. State the reason when it is not obvious.
- Do not repeat the same claim at unit, integration, and differential layers. An additional
  layer must catch a different bug. Reuse existing harnesses and fixtures. A refactor that preserves
  behavior usually needs the existing regressions to pass, not a new test for each extracted helper.
- Use small named case tables in Rust or pytest parametrization only for meaningful variation.
  Delete equivalent cases; do not compress a redundant suite into a large table. Keep separate
  tests when setup or failure meaning differs. Avoid Cartesian products without a concrete need.

Useful targets include a previously broken command, a public API round trip, a guest-visible
state transition, a wire format used by an external consumer, and a quota or isolation boundary
whose failure has an observable consequence. Choose inputs that expose the failure rather than
enumerating every input shape.

For example, a regression for a broken `head` pipeline should execute that pipeline and check its
result through the existing command harness. It should not separately prove that a `HeadOptions`
type exists, that its constructor stores `count`, and that a dispatcher calls a helper.

## Reject tests that do not earn their keep

- Type or symbol existence, trait guarantees already checked by Rust, constructor field
  assignments, constant values, or getters returning the value just set.
- Tests that exercise the host language or a dependency rather than shellsim's behavior.
- A case for each scalar/configuration guard without a reported regression, public compatibility
  contract, or simulation safety consequence.
- Internal helper call counts, dispatch checks, private layout, or source-text searches used as
  substitutes for behavior. Private algorithm invariants may justify a focused unit test.
- Expected-value helpers that copy the production algorithm, tautologies, swallowed failures,
  or assertions accepting unrelated outcomes.
- Smoke probes that only establish construction, registration, or successful execution. Use
  disposable probes during development; check in a behavioral assertion or remove the probe.
- Configuration projections: setting a field to a value and asserting that a lowered or copied
  field equals that value. Exercise the resulting behavior, a meaningful merge conflict, or a
  real external wire translation instead.
- Assertions on incidental log messages, error wording, defaults, or dependency versions.
  Exact text belongs in a test only when it is the compatibility contract.
- Permanently ignored tests, empty test files, and screenshots without behavioral assertions.
  A temporary exclusion needs a concrete reason and a condition for restoring the test.

In Rust, these assertions usually add no protection:

```rust
assert!(matches!(mode, Mode::Read)); // The test just assigned Mode::Read.
assert_eq!(options.limit, 10);      // The constructor just received 10.
assert!(result.is_ok());           // The intended output and effects remain unchecked.
```

The first two repeat setup. The third is sufficient only when successful completion itself is
the contract, such as a regression for a panic on a particular input; make that reason clear.
Likewise, `is_err()` alone is insufficient when an unrelated failure would pass. Check the
relevant error variant and any state that must remain unchanged. Do not add redundant assertions
just to make a test look more thorough.

Command construction is rarely the contract. Run the command through the boundary and check its
effect rather than searching rendered command text for a flag. When an external command protocol
cannot reasonably be exercised, assert parsed arguments and values, with a reason for that boundary.
For structured output, compare the fields consumers use. Reserve exact text fixtures for shell,
Python, serialization, or other compatibility surfaces where bytes matter.

## Mocks, fixtures, and test structure

Prefer real execution and existing in-memory fixtures. Fake an external boundary when needed;
do not mock internal collaborators merely to prove wiring. Do not build a test framework for a
single case.

Use shellsim's environment, virtual filesystem, and clock to observe effects. Product code must
remain capability-free; a host subprocess used as a reference belongs only in the test harness.
For host adapters, use a temporary directory or a boundary fake when real I/O would be slow,
destructive, or nondeterministic. Assert resulting state rather than the number of helper calls.

Keep setup small enough that the failing behavior is apparent. Reuse a helper when it removes
repeated mechanics, but keep the scenario and expected result visible in the test. Avoid generic
fixture builders, new dependencies, or macros for a handful of straightforward cases.

Use ordinary Rust `#[test]` functions and named case tables with a useful case label on failure.
Keep warranted private unit tests in the module's `#[cfg(test)]` area or existing `tests.rs`.
For host Python tooling, prefer top-level pytest functions, fixtures, and parametrization over
test classes. Follow the existing portable-suite conventions for guest Python tests. Extend the
existing suite before creating another integration-test crate or harness.

## Preserve simulator contracts

Guest language semantics are shellsim behavior, so testing them is appropriate. Assert Python
exception types and guest-visible value types when compatibility depends on them; an assertion
that merely repeats a host type annotation is not useful coverage.

Keep regressions for isolation, resource accounting, and unsupported behavior when a change can
break those boundaries. Assert the observable refusal, exhaustion, or lack of side effects;
do not duplicate every validation branch. A test that reaches a limit should demonstrate bounded
behavior under a small budget, not consume a large amount of host memory to prove exhaustion.

Use virtual time and explicit state transitions. Do not use host sleeps, elapsed-time thresholds,
or retry loops to make a race pass. Keep inputs and expectations independent of host locale,
network state, ambient filesystem contents, and unordered collection traversal.

Numerical regressions need an independent, simple reference and a small set of meaningful inputs.
Choose tolerances for the operation and dtype; do not copy the production calculation into the
expected-value helper. Do not widen tolerances or weaken expectations to make a failure pass.
Changing a tolerance requires explicit user agreement and an explanation of the intended
numerical contract. Preserve checks of shape, dtype, exceptions, or warnings when compatibility
depends on them, without expanding every case into a full Cartesian product.

## Review and validation

For each added test, ask:

- What distinct bug would fail this test, and is it already caught elsewhere?
- Does the assertion observe the claimed behavior, or only repeat setup or implementation?
- Could the behavior be wrong while this test still passes?
- Would a harmless refactor break the test?
- If this is a unit test, why is a regression through an existing interface impractical?
- Do its fixtures, cases, and runtime cost more than the protection they provide?

Ask for low-value tests to be deleted or rewritten. Do not require a written
justification on every test, and do not delete unrelated coverage as part of a focused change.

Run a narrow relevant test while editing and the repository gates before handing off a code
change, as described in `AGENTS.md`. Fix failures caused by the change. A formatter-only follow-up
needs formatting and lint checks again; repeat tests when executable behavior or expectations
change. Do not rerun a passing suite merely to accumulate validation evidence.
