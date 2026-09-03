# TaskTrove stdlib implementation matrix

This is a static, reproducible API inventory for the extracted
`open-thoughts/OpenThoughts-TBLite` tree at commit `7b70111339b4af23cece95d63aeec1c705790868`.
The local mirror contains 100 task directories (plus an `assets/` directory); 97 import at least
one reviewer-named module.
The matrix reads Python files and Python heredocs in reference `solution/*.sh` files with the
standard-library-only tool [`tasktrove_stdlib_matrix.py`](../tools/tasktrove_stdlib_matrix.py).
It never executes a task or extracts an archive.

The checked-in machine-readable reports contain counts and representative task/source paths:

- [`TASKTROVE_STDLIB_MATRIX.json`](TASKTROVE_STDLIB_MATRIX.json)
- [`TASKTROVE_STDLIB_MATRIX.tsv`](TASKTROVE_STDLIB_MATRIX.tsv)

The tool's default JSON/TSV mode retains every task and source path. `--compact` (used for the
checked-in reports) retains counts plus five task examples and three source examples per row.
Both formats are sorted by module and API. Reproduce and validate the reports with:

```sh
python3 tools/tasktrove_stdlib_matrix.py /tmp/openthoughts-tblite-7b70111 --format json --compact
python3 tools/tasktrove_stdlib_matrix.py /tmp/openthoughts-tblite-7b70111 --format tsv --compact
```

## Module order

Stage numbers are implementation order, not a claim that every method in a module belongs in the
first release. “Safe native” means a deterministic host computation behind a narrow Rust API;
“VFS wrapper” means all effects go through shellsim's virtual filesystem/process capability;
“pure shim” means an in-emulator implementation; “unsupported” is reserved for APIs whose
semantics would escape the capability boundary.

| Stage | Module | Tasks | Classification | High-value observed surface |
|---:|---|---:|---|---|
| 1 | `json` | 75 | pure shim | `load`, `loads`, `dump`, `dumps`, `JSONDecodeError` |
| 1 | `os` | 69 | VFS wrapper | `makedirs`, `path.exists`, `path`, `path.join`, `access`, `listdir`, `environ` |
| 1 | `typing` | 57 | pure shim | `Any`, `Dict`, `Optional`, `List`, `Tuple`, `Callable`, `Iterable`, `Sequence`, `Set`, `Union` |
| 1 | `sys` | 26 | safe native | `path`, `argv`, `exit`, `executable`, `stderr`, `stdout`, `stdin` |
| 1 | `collections` | 16 | pure shim | `defaultdict`, `Counter`, `deque` |
| 2 | `re` | 30 | pure shim | `search`, `match`, `findall`, `compile`, `escape`, `sub`, flags |
| 2 | `math` | 6 | safe native | `log`, `sqrt`, `exp`, `ceil`, `sin`, `isinf`, `isnan` |
| 2 | `argparse` | 6 | pure shim | `ArgumentParser`, `Namespace` |
| 2 | `string` | 2 | pure shim | `ascii_lowercase` |
| 2 | `dataclasses` | 2 | pure shim | `dataclass` |
| 2 | `enum` | 1 | pure shim | `Enum` |
| 2 | `itertools` | 1 | pure shim | module import only in this sample |
| 2 | `heapq` | 1 | pure shim | `heappush`, `heappop` |
| 2 | `bisect` | 0 | pure shim | reviewer-requested; add conformance probes |
| 2 | `functools` | 0 | pure shim | reviewer-requested; add conformance probes |
| 3 | `subprocess` | 32 | VFS wrapper | `run`, `Popen`, pipes, `check_output`, timeout/error types |
| 3 | `pytest` | 62 | pure shim | `fail`, `fixture`, `skip`, `raises`, `main`, `mark` |
| 3 | `unittest` | 0 | pure shim | absent from this sample; add conformance probes |

## API details and boundaries

The JSON and TSV reports carry exact access and call counts for every observed attribute, constant,
class, and direct `from ... import ...` binding. Representative rows include:

| Module | API | Tasks / accesses / calls | Representative provenance |
|---|---|---:|---|
| `json` | `load`, `dump` | 41 / 270 / 270; 59 / 127 / 127 | `application-debug/tests/test_outputs.py`; `anomaly-detection-ranking/solution/solve.sh` |
| `os` | `makedirs`, `path.exists` | 47 / 90 / 90; 25 / 88 / 88 | `anomaly-detection-ranking/tests/test_outputs.py`; `bandit-delayed-feedback/tests/test_outputs.py` |
| `collections` | `defaultdict`, `Counter`, `deque` | 10 / 105 / 105; 2 / 4 / 4; 4 / 7 / 2 | `build-system-task-ordering/solution/solve.sh`; `book-portfolio-analysis/tests/grader.py` |
| `re` | `search`, `match`, `findall` | 15 / 51 / 51; 12 / 31 / 31; 5 / 12 / 12 | `application-debug/tests/test_outputs.py`; `cpp-daemon-sighup-segfault/tests/test_outputs.py` |
| `subprocess` | `run`, `Popen` | 29 / 135 / 135; 7 / 8 / 8 | `acl-permissions-inheritance/tests/test_outputs.py`; `basic-message-queue/tests/grader.py` |
| `pytest` | `fail`, `fixture`, `skip` | 51 / 134 / 134; 10 / 22 / 9; 3 / 15 / 15 | `anomaly-detection-ranking/tests/test_outputs.py`; `bandit-delayed-feedback/tests/test_outputs.py` |
| `typing` | `Dict`, `Any`, `Optional` | 49 / 226 / 0; 46 / 114 / 0; 46 / 205 / 0 | `anomaly-detection-ranking/tests/grader.py` |

Counts are `tasks / syntactic accesses / calls`; a call can be zero for a constant, exception type,
or annotation marker. Full provenance is available from the tool without `--compact`.

The following observed names should be explicit negative tests or capability-gated wrappers rather
than silently implemented as ordinary filesystem helpers:

- `os.getpgid`, `os.kill`, `os.killpg`, `os.setsid`, and `os.system` (host process and signal
  control); `os.chdir` is only safe when mapped to the VFS working directory.
- `subprocess.Popen` lifecycle, signals, arbitrary executable paths, and long-lived processes;
  synchronous `run`/`check_output` can be routed to the shellsim dispatcher for Python and known
  shellsim commands, with `check=True`, timeout, pipes, and error objects modeled explicitly.
- `pytest.mark.asyncio` and async fixtures (one observed task) remain unsupported until async VM
  semantics are intentionally added. `pytest` should be a native runner facade with common flags,
  not an embedded third-party plugin system.

## Staged validation

1. Add focused CPython differential scripts for every observed API row in stages 1 and 2. Compare
   stdout, stderr, exit status, and normalized JSON; reject nondeterministic values rather than
   masking them.
2. Use the provenance paths to select roughly 100 small deterministic scripts across task,
   environment, test, and solution sources. Keep each script standalone or provide a VFS module
   fixture, and record the source task and source path in the test name.
3. Run the selected scripts on host CPython and shellsim's `python3.14` under identical argv,
   environment, and VFS contents. A timeout or capability rejection is a visible result, never a
   pass.
4. Promote a task to acceptance only after its complete pure-Python wrapper and test behavior
   matches; retain the module probes as regression tests so later task additions cannot hide API
   drift.
