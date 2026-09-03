# Minimal Python test-runner design

This note narrows the `pytest`/`unittest` item in `PYTHON_3_14_PROPOSAL.md` to a safe,
auditable compatibility runner. It is not a plan to embed upstream pytest, load plugins, or
implement arbitrary test-framework reflection.

## Evidence from the codebase and corpus

The current interpreter has a custom lexer/parser/compiler/VM and VFS-backed imports. The first
pytest slice now exposes `python -m pytest FILE`, the shell `pytest FILE` launcher, and the same
VFS-only runner behind both entry points. It collects top-level zero-argument `test_*` functions
in source order, executes each through the VM, and returns a nonzero status for failures or zero
tests. Fixtures, decorators, async tests, and unknown flags remain explicit unsupported paths;
an unknown runner feature must not be reported as a passing test run.

The historical implementation at commit `1e06916` used a RustPython prelude plus an embedded
`driver_pytest.py`. That driver is useful as a behavioral sketch, not as a drop-in implementation:
it collected top-level `test_*` functions and `Test*` methods in definition order, resolved a
small fixture set, advanced generator fixtures for teardown, and returned a nonzero status for
zero tests. Its `capsys` implementation was a no-op, `monkeypatch` mutation was incomplete, and
all teardown exceptions were swallowed. Those shortcuts are unsafe for this VM and should not be
reintroduced.

The checked-in TaskTrove inventory reports 62 tasks importing pytest and no task importing
unittest. Observed pytest APIs include `fail`, `fixture`, `skip`, `raises`, `main`, `mark`, and
`param`; one task uses `pytest.mark.asyncio` and async fixtures. The fixture-bearing examples are
concentrated in ordinary `tests/test_*.py` files (for example `todos-api`,
`bloom-filter-cache-penetration-prevention`, and `multi-labeller`). The `multi-labeller` fixture is
module-scoped. A separate fixture probe should add unittest despite its absence in this sample.

## Proposed module split

Keep runner code behind the Python capability boundary and make it consume VM interfaces rather
than `Environment` directly:

* `src/python/testing/mod.rs` owns `TestPlan`, `TestItem`, `FixtureSpec`, `TestOutcome`, and the
  shared collection/invocation protocol. It owns no host filesystem or process calls.
* `src/python/testing/collect.rs` walks the VFS through the platform capability, selects stable
  `test_*.py` and `*_test.py` paths, imports each module once, and records source-definition
  ordering using source spans. It collects top-level `test_*` callables and `Test*` classes with
  `test_*` methods. No `dir()` sorting is used where source order is available.
* `src/python/testing/fixtures.rs` registers decorator metadata and built-in fixtures. It resolves
  dependencies as a cycle-checked graph, creates scopes (`function`, `class`, `module`, `session`),
  and stores a finalizer stack. A generator fixture is split at its first `yield`; the remainder
  runs exactly once in reverse dependency order. Finalizer failures become test/session errors,
  never silently disappear.
* `src/python/testing/assertions.rs` contains runner values and exceptions for `raises`, `approx`,
  `fail`, `skip`, and `xfail`. `raises` works both as a context manager and as a callable form.
  `approx` delegates comparison to VM numeric/equality protocols and has bounded recursion.
* `src/python/testing/pytest.rs` installs a tiny `pytest` module (`fixture`, `mark`, `param`,
  `raises`, `approx`, `fail`, `skip`, `xfail`, `main`) and translates CLI arguments into a
  `TestRequest`. It is a facade over the shared runner, not a second test engine.
* `src/python/testing/unittest.rs` installs `unittest.TestCase`, common assertion methods,
  `setUp`/`tearDown`, `setUpClass`/`tearDownClass`, `TestSuite`, and a small `TextTestRunner`.
  Discovery and execution still use the shared plan and outcome types.
* `src/python/testing/report.rs` emits deterministic concise output and exit status. It keeps
  stdout/stderr capture separate from result reporting and records skipped, xfailed, errors,
  unsupported capabilities, and teardown failures.

The runner should receive a narrow capability object such as `TestPlatform` with VFS listing/read,
temporary-path creation, environment overlay, and bounded output capture. It must not call
`std::fs`, `std::process`, wall-clock time, sockets, or host environment APIs. Each test invocation
gets fresh locals and a fixture scope context but shares only the explicitly selected VFS/module
scope. A failed test cannot leak a fixture mutation into a later function-scoped test.

## Exact first slice

The first slice is implemented and covered by `tests/python_pytest.rs`. It deliberately keeps
the report format stable and local to shellsim (`PATH::test_name PASSED|SKIPPED|FAILED ...`) rather
than claiming byte-for-byte pytest output compatibility. CPython differential checks compare the
assertion failure shape and the runner's collection/order/status semantics. The stdlib probe is
promoted only for the import-only pytest case because the focused runner tests also pass.

Implement this order, with one behavior promoted only after a CPython comparison:

1. Shared result types and a VFS-only collector for one explicit file. Support stable path and
   definition order, top-level functions, and zero-test failure.
2. `pytest.fail`, `pytest.skip`, `pytest.raises` (context-manager form), `pytest.approx`, and the
   `pytest.fixture` decorator with function scope. Unknown fixture names are collection errors.
3. Dependency injection for function fixtures plus reverse-order generator teardown. A teardown
   error is reported and makes the run fail.
4. Module-scoped fixtures and built-ins `tmp_path`, `monkeypatch`, and `capsys`. `tmp_path` is a
   VFS path; `monkeypatch` maintains an undo log for environment, VFS cwd, attributes, and items;
   `capsys.readouterr()` drains the per-test buffers. Do not provide a fake object for an unknown
   fixture.
5. `mark.parametrize` with literal values and deterministic IDs, then class collection and the
   common unittest assertion/setup slice.
6. Wire `python -m pytest`, `pytest`, and `python -m unittest` to the same runner. Accept only the
   observed flags (`-q`, `-v`, `-rA`, `--tb=short`, and explicitly handled `--ctrf`); reject all
   other flags with a visible diagnostic. `--ctrf` should either produce a bounded deterministic
   report or fail explicitly until report writing is implemented.

Async functions/fixtures, `pytest_asyncio`, plugin loading, assertion rewriting, subprocess
workers, xdist, coverage, arbitrary `conftest.py` hooks, and dynamic collection are explicit
frontier features. In particular, `pytest.mark.asyncio` must be rejected as unsupported rather
than running a coroutine object as if it were a normal test.

## Collection, execution, and teardown contract

The runner should follow this sequence:

```text
CLI -> TestRequest -> VFS discovery -> import module once
    -> collect decorators/classes/fixtures in source order
    -> resolve selected node IDs and parametrization
    -> create session/module/class scopes
    -> for each item: function fixture setup -> invoke -> capture -> finalize
    -> finalize class/module/session scopes -> deterministic report + exit status
```

Collection errors stop before executing test bodies. A fixture cycle, missing fixture, duplicate
parameter ID, unsupported async object, or import failure is an error with its source location.
`skip` is a reported skip; `xfail` is only an expected failure when the test actually fails;
unexpected passes are reported separately. Exit status is zero only when every selected item is
pass or expected-failure and all finalizers pass, and at least one item was collected.

The default output should avoid traceback paths that vary between host and VFS. The differential
contract compares exit status, captured stdout, captured stderr, and a normalized outcome table;
diagnostic traceback text is tested separately by shape (exception type and source location), not
as an opaque byte-for-byte host string.

## Validation plan

Add standalone fixtures under `tests/fixtures/python/testing/` and run each unchanged through
CPython 3.14 and shellsim. Start with:

* function order and zero-test behavior;
* `pytest.fail`, `skip`, `raises`, and `approx` pass/fail cases;
* fixture dependency order, function-scope isolation, module-scope caching, and generator
  teardown on pass and on exception;
* unknown/cyclic fixture diagnostics;
* `tmp_path` VFS isolation, `monkeypatch.setenv`/`delenv`/`chdir` undo, and `capsys` drain;
* literal `mark.parametrize` expansion and stable IDs;
* unittest `TestCase` assertions, setup/teardown, class hooks, suite order, and no-tests status;
* runner entrypoint/flag handling and explicit rejection of async/plugin/host-process cases.

The existing 100-script TaskTrove corpus remains a language/stdlb differential suite, not a test
runner suite. To prevent false confidence, runner fixtures should have their own manifest recording
source shape, expected CPython 3.14 status/output, and whether shellsim supports or rejects the
case. Promote real TaskTrove verifier files only when their complete VFS package and test runner
behavior matches; a reduced test file is useful for development but is not an acceptance claim.

## Safe next implementation milestone

The current VM can support the first collector only after it exposes callable metadata/source
spans, deterministic VFS listing, and a structured exception result to the runner. The next safe
coding milestone is therefore shared `TestOutcome`/collection metadata plus `pytest.fail` and
function fixtures, followed by the focused differential fixtures above. It is not safe to claim
upstream pytest or the 23 pure-package tasks until module imports, VFS fixture effects, teardown,
and exit/report semantics have all been compared against CPython.
