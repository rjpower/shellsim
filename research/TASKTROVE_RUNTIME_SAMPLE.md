# TaskTrove Python runtime sample

This note measures how unchanged Python from a small TaskTrove-derived corpus reaches shellsim.
It supersedes the optimistic interpretation of the reduced 100-script corpus. That fixture is
still useful as a regression suite, but its source reductions remove syntax and dependencies that
are material in real tasks.

## Sample and method

The sample is the 100-task OpenThoughts-TBLite checkout at commit
`7b70111339b4af23cece95d63aeec1c705790868`. TBLite is a curated subset intended for agent
iteration and debugging. Of those tasks, 49 contain Python in a reference solution and 97 contain
Python verifier tests.

`tools/tasktrove_runtime_probe.py` extracts direct `.py` files and likely Python heredocs from
reference solutions without executing their shell wrappers. It runs each payload unchanged in an
isolated temporary project through `shellsim-python`. It also mounts each task and asks shellsim's
pytest runner to collect its unmodified `test_*.py` verifier files.

The result is source loadability, not a task solve rate. Each run records only its first blocker.
Removing one blocker can reveal another, and reference solutions often assume services, installed
packages, and files that do not exist in a standalone invocation.

Reproduce the sample after building the release runner:

```sh
cargo build --release --bin shellsim-python
python3 tools/tasktrove_runtime_probe.py /path/to/OpenThoughts-TBLite
```

## Result

Only 1 of 69 reference-solution Python payloads loaded unchanged. This was 1 of the 49 tasks with
solution Python. None of the 97 verifier groups reached a passing collection and execution.

| First blocker | Solution payloads | Verifier groups |
| --- | ---: | ---: |
| Triple-quoted strings | 30 | 90 |
| Bitwise operators | 8 | 1 |
| Formatted/raw f-string forms | 7 | 1 |
| Comment before first suite statement | 5 | 0 |
| Conditional expressions | 3 | 0 |
| Async functions | 2 | 0 |
| Slices | 2 | 0 |
| Explicit line continuation | 1 | 4 |
| Relative import syntax | 1 | 0 |
| Missing modules | 9 | 0 |
| Other indentation behavior | 0 | 1 |

The missing-module first blockers were `pandas` (3 payloads), then one each for `cryptography`,
`datetime`, `glob`, `hashlib`, `mlflow`, and `uuid`. They undercount module gaps because parsing
and earlier imports stop most payloads first.

## Follow-up after first-order compatibility work

Rerunning the same checkout after the first-order parser and frozen-stdlib pass yields 2 of 69
solution payloads, representing 2 tasks, and still 0 of 97 verifier groups. The small change in
whole-payload passes is expected: the probe executes each standalone solution source, so successful
parsing now exposes absent task files, third-party packages, and later language constructs.

The original dominant syntax blockers no longer appear as first blockers: triple-quoted strings,
formatted/raw f-strings, conditional expressions, comment-first suites, relative imports, and the
common explicit-continuation cases all dropped out of the solution result. The remaining named
syntax categories are four async payloads, two multidimensional scientific-array slices, and one
bitwise-classified payload requiring a still-unsupported operator form. Missing modules are now
visible as the main bounded-runtime frontier, led by `pandas` (9), `datetime` (7), `numpy` (4), and
two `socket` payloads. The new `glob`, `hashlib`, `pathlib`, and `uuid` modules are no longer import
blockers.

Verifier collection remains dominated by task harness integration: 40 of 97 groups first fail on
the task-provided `grader` package. The next useful work is richer call and parameter grammar,
pytest fixtures/decorators, import-root mapping, and small deterministic modules such as `datetime`
and `tempfile`. The result does not justify emulating NumPy, pandas, network services, or other
native ecosystems inside the interpreter.

## What to implement next

The highest-leverage work is language compatibility, in this order:

1. Lex triple-quoted strings and docstrings. This is the first blocker for 30 of 69 solution
   payloads and 90 of 97 verifier groups.
2. Complete common expression grammar: bitwise `&`, `|`, and `^`; slices; conditional
   expressions; raw f-strings; and explicit backslash continuation.
3. Make suite parsing ignore leading blank and comment-only lines. This is ordinary generated
   Python and should not require a runtime abstraction.
4. Expand collection only after those sources parse: pytest fixtures and decorators, `tmp_path`,
   common runner flags, and package-aware discovery are likely next blockers.
5. Add deterministic, modeled slices of `pathlib`, `datetime`, `hashlib`, `csv`, `glob`, `random`,
   `tempfile`, `io`, and `uuid`. They fit shellsim's capability model and recur in the static
   inventory. Preserve VFS, virtual-clock, and deterministic-random boundaries.
6. Improve project mapping and imports for task layouts that assume `/app`, packages, and relative
   imports. Any convenience mapping must remain an explicit harness choice, not ambient host
   access.

Large native ecosystems and service workloads should remain explicit frontiers for now. `pandas`,
NumPy, scikit-learn, MLflow, cryptography, databases, HTTP servers, and async service frameworks
would each dwarf the small deterministic stdlib slices. A safe nested-Python subprocess facade
may eventually unlock verifier helpers, but it must invoke only shellsim's interpreter with copied
VFS state and shared resource accounting.

The immediate lesson is that broad, ordinary parser coverage has more value than another deep
object-model feature for this corpus. Once the parser items above land, rerun the same first-blocker
probe rather than projecting a pass rate from these counts.

## Follow-up after source stdlib and byte values

The September 11, 2026 resample after source-backed stdlib modules and byte-preserving `bytes` and
`bytearray` passes 7 of 69 unchanged solution payloads, representing 6 tasks whose solution Python
all passes. The pre-bytes sample passed 4 payloads and 3 complete tasks. This remains a source
loadability measure, not a task solve rate.

`base64`, `codecs`, `struct`, `tempfile`, and binary-mode I/O no longer appear as first import or
representation blockers. The two `struct`-using reverse-engineering payloads now reach their
expected missing input files. The remaining solution failures are led by third-party ecosystems
(`pandas` 10, async syntax 5, NumPy 4, and `socket` 3), followed by task files that are intentionally
absent from an isolated payload invocation. Small actionable API gaps include argparse subparsers,
the fail-closed subprocess facade, and richer parameter and subscription grammar.

Verifier task passes remain 0 of 97. Forty verifier groups still stop first on the task-provided
`grader` package, while others require pytest decorators, subprocess execution, importlib, async,
or syntax outside the current slice. The bytes work removes a shared correctness obstacle, but it
does not change those harness boundaries.
