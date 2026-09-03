# TaskTrove Python inventory

Status: Phase 0 evidence, 2026-09-02

## Provenance and method

This inventory uses the public Apache-2.0 TBLite data in two forms:

- `NousResearch/openthoughts-tblite` commit
  `44c975f590dde88316572d7e2a779ec1112d4a4b`, whose single 100-row Parquet has LFS SHA-256
  `ca87c91685ad768e108404906c6253497b149bb3e8da75a6186069b40574d698`; and
- `open-thoughts/OpenThoughts-TBLite` commit
  `7b70111339b4af23cece95d63aeec1c705790868`, which includes the same environments and tests plus
  public `solution/solve.sh` reference solutions.

The Parquet contains separate base64-encoded gzip-tar environment and test columns. The inventory
tool reads archives in memory, rejects absolute and parent-traversal paths, does not follow archive
links, caps individual members at 4 MiB and a task at 64 MiB, and never extracts to disk. It also
recognizes likely Python heredocs in reference `solve.sh` files and parses them with CPython 3.14's
`ast` module.

The mirror-to-source comparison checked 1,537 files with no missing paths. All source, test,
configuration, and shell files matched. Twenty-four large binary/LFS files compared against Git LFS
pointer contents in the shallow source checkout rather than hydrated binary contents; three data
files were intentionally skipped by the analysis size cap. None affects the syntax/API inventory.

The checked-in tool is `tools/tasktrove_inventory.py`. Reproduce the two reports with:

```sh
uv run --no-project --with pyarrow \
  python tools/tasktrove_inventory.py /path/to/tasks.parquet > inventory.json

python3.14 -B tools/tasktrove_inventory.py /path/to/OpenThoughts-TBLite \
  > inventory-with-solutions.json
```

The current TaskTrove schema has only `path` and `task_binary` columns; the tool supports that form
as well. TaskTrove's full default config is currently a multi-source 9.32 GB collection, so the
bounded 100-task Parquet was used instead of downloading the entire repository.

## What was observed

Across the 100 tasks, the tool parsed 268 Python sources, including likely Python heredocs from 48
reference solutions. All 100 tasks contain Python somewhere in the environment, verifier, or
solution, and none of the inspected sources had a Python 3.14 syntax error.

### Language features

| AST feature | Tasks | Occurrences |
|---|---:|---:|
| Functions | 99 | 1,663 |
| `with` | 86 | 775 |
| `try` | 76 | 399 |
| Classes | 61 | 138 |
| List comprehensions | 53 | 295 |
| Generator expressions | 51 | 258 |
| Lambdas | 24 | 79 |
| Dict comprehensions | 20 | 95 |
| `global` | 8 | 13 |
| `yield` | 7 | 14 |
| Async functions | 6 | 62 |
| `nonlocal` | 3 | 5 |
| `await` | 2 | 47 |
| Walrus expressions | 1 | 12 |
| `match` | 0 | 0 |

This validates the proposal's emphasis on scope analysis, closures, exceptions/unwinding,
descriptors/classes, comprehensions, context managers, and generators. A scalar expression engine
plus library shims cannot cover this corpus.

Async code is concentrated in service-oriented tasks with FastAPI/Redis/httpx dependencies. It can
remain outside the initial pure, in-process slice unless the reviewer-provided 23-task membership
includes one of those tasks.

### Requested modules

These counts include environment code, tests, and recognized reference-solution Python:

| Module | Tasks importing it | Directly observed API examples |
|---|---:|---|
| `json` | 75 | `load`, `loads`, `dump`, `dumps`, `JSONDecodeError` |
| `os` | 69 | `environ`, cwd/files/dirs/walk/path, access checks; process APIs occur only in non-pure cases |
| `typing` | 57 | `Any`, `Dict`, `List`, `Optional`, `Tuple`, plus a few `Callable`/`Iterable`/`Sequence`/`Set`/`Union` |
| `subprocess` | 32 | `run`, `Popen`, `check_output`, pipes/errors/timeouts |
| `re` | 30 | compile/search/match/fullmatch/findall/finditer/sub/escape, `IGNORECASE`, `MULTILINE` |
| `sys` | 26 | argv/path/stdio/exit/executable/modules/version |
| `collections` | 16 | `defaultdict`, `deque`, `Counter` |
| `argparse` | 6 | `ArgumentParser`, `Namespace` |
| `math` | 6 | ceil/exp/isinf/isnan/log/sin/sqrt |
| `string` | 2 | `ascii_lowercase` |
| `dataclasses` | 2 | `dataclass` |
| `enum` | 1 | `Enum` |
| `itertools` | 1 | imported by a combinatorics task |
| `heapq` | 1 | `heappush`, `heappop` |
| `bisect` | 0 | reviewer-requested, not exercised here |
| `functools` | 0 | reviewer-requested, not exercised here |

The zero/low counts do not remove reviewer-requested modules from scope. They do tell us where a
small conformance suite, rather than TBLite alone, must define the API contract.

### Additional modules required by the measured corpus

The original proposal treated these as possible import dependencies. The sample shows that several
must be promoted into the first compatibility tier:

| Module | Tasks |
|---|---:|
| `pathlib` | 61 |
| `datetime` | 17 |
| `time` | 17 |
| `hashlib` | 12 |
| `random` | 10 |
| `csv` | 9 |
| `glob` | 7 |
| `sqlite3` | 7 |
| `tempfile` | 4 |
| `importlib.util` | 4 |
| `threading` | 4 |
| `io`, `logging`, `struct`, `uuid` | 3 each |
| `asyncio` | 2 |

For the initial pure-package slice, promote `pathlib`, `datetime`, `time`, `hashlib`, `random`,
`csv`, `glob`, `tempfile`, `importlib.util`, and `io`. Treat `sqlite3`, real threading, and asyncio
as separate capability decisions because they introduce substantial semantics beyond simple helper
functions.

### Test runners

- 97 of 100 `tests/test.sh` scripts invoke pytest.
- 62 tasks import `pytest` in Python; none imports `unittest`.
- Observed pytest Python APIs are mostly `fail` (51 static references), `fixture` (10), `skip` (3),
  `main` (2), and one each of `raises`, `mark`, and `mark.asyncio`.
- Observed built-in fixture arguments include `tmp_path` plus task-local fixture graphs. Four
  service tasks use a `client` fixture; the remainder are mostly one-off task fixtures.
- 29 runner scripts request the third-party `--ctrf` option, 96 use `-rA`, 11 use `-q`, 11 use
  `-v`, and 3 use `--tb=short`.

The native pytest adapter should therefore accept the common reporting flags and either generate
the expected CTRF artifact or record that the option is unsupported. It should not attempt to run
upstream pytest or its plugin system. `unittest` needs its own reviewer-driven conformance suite,
because TBLite provides no coverage for it.

### Subprocess boundary

Thirty-one tasks contain statically recognizable subprocess calls. They divide cleanly:

- safe nested wrappers such as `[sys.executable, script]`, `python file.py ...`, or `python -c ...`;
- shellsim commands such as `bash`, `cat`, `sha256sum`, `jq`, and `git`; and
- genuinely out-of-domain processes such as Java/Maven, Node/npm, native validators, systemd,
  OpenSSL, servers, Redis, and long-lived process/signal tests.

The first two groups can route synchronously through the shared shellsim dispatcher. The third must
remain unsupported rather than being made to look successful. `Popen` occurs in both groups, so a
small completed-on-construction `Popen`/`communicate` facade is useful for Python wrappers, but it
must reject lifecycle, concurrency, signal, and server behavior.

Representative Python-wrapper shapes found in the corpus include:

```text
subprocess.run([sys.executable, str(script_file)], capture_output=True, text=True, timeout=...)
subprocess.run(["python", "/app/merkle_cli.py", "scan", path], capture_output=True, text=True)
subprocess.run(["python3", script_path], cwd=..., check=True)
subprocess.run(cmd, capture_output=True, text=True)
subprocess.Popen(["/app/bandit", "--seed", str(seed)], stdin=PIPE, stdout=PIPE, text=True)
```

The last shape invokes a task-produced executable and is not a Python wrapper; it stays outside the
safe Python slice unless shellsim has a faithful native command model for that executable.

## Representative samples

These are not asserted to be the reviewer's exact 23-task bucket; neither the Parquet schema nor
the task metadata contains a `pure packages` label. They are useful vertical slices selected from
the public corpus:

| Task | Why it is informative |
|---|---|
| `build-system-task-ordering` | Pure multi-file algorithm; classes, closures/nonlocal, lambdas, comprehensions, loops; `collections`, JSON, VFS `importlib.util` |
| `raft-log-repair-concurrent-access` | Pure VFS-imported solution with functions, exceptions, comprehensions, and import isolation |
| `python-api-rate-limit` | Multi-file package, classes, annotations, collections, datetime, JSON, random, and nested Python subprocess execution |
| `build-merkle-tree-cli-sha512` | Argparse CLI invoked repeatedly through subprocess; files, hashing, JSON, exceptions, context managers |
| `malicious-package-forensics` | Object-heavy processing with CSV/datetime/JSON/regex/pathlib and many context managers |
| `monorepo-changelog-cli` | Pytest fixture graph, `tmp_path`, regex/collections, and nested Python/shell command wrappers |
| `multi-labeller` | Dataclass + argparse package, generator fixture teardown, local imports, file mutation; also exposes a NumPy boundary |
| `service-deployment-wave-planner` | Large deterministic pure computation: classes, nested comprehensions/generators, datetime, JSON, hashing, VFS paths |
| `submission_a63937a5_20251224_152124` | Argparse/stdin CLI, JSON, hashing/HMAC, math/random, and a Popen protocol boundary |
| `fix_async_worker_queue` | Deliberate negative boundary sample: enum plus async/await, FastAPI/httpx/pydantic/uvicorn and client fixtures |

The first implementation milestone should use `build-system-task-ordering` as the complex semantic
target and a smaller extracted scalar/function fixture as the bootstrap target. The nested CLI
milestone should use `build-merkle-tree-cli-sha512`.

## Resulting plan adjustments

1. Keep the custom parser/compiler/VM design; the measured AST breadth confirms it is foundational.
2. Promote the common additional stdlib modules listed above instead of waiting for incidental
   import failures.
3. Implement VFS source imports and `importlib.util.spec_from_file_location` early; two historically
   successful tasks depend on it directly.
4. Implement synchronous nested Python subprocesses after script execution but before broad stdlib
   work, using the exact wrapper shapes above.
5. Keep async, real process lifetimes, services, SQLite, and threading behind explicit capability
   gates until the exact 23-task list proves one is required.
6. Add pytest fixture teardown and `tmp_path` before broad pytest API emulation; those behaviors are
   observed, while upstream plugin compatibility is not.
7. Obtain the reviewer-authored 23 task IDs before creating the acceptance manifest. Sampling can
   shape architecture, but it cannot recover a label that is absent from the source data.

## Public source references

- [TaskTrove dataset card](https://huggingface.co/datasets/open-thoughts/TaskTrove) — aggregate
  layout and `path`/gzip-tar `task_binary` schema.
- [TBLite flat Parquet mirror](https://huggingface.co/datasets/NousResearch/openthoughts-tblite) —
  100-row schema and unchanged environment/test archive claim.
- [Original OpenThoughts-TBLite tree](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite/tree/main)
  — public task directories and reference solutions.
