# TaskTrove workflow sample

This sample identifies likely shellsim support gaps by running complete TaskTrove golden solutions
and verifiers. It is a prioritization aid, not a task pass-rate benchmark.

## Method

The sample uses the 100-task OpenThoughts-TBLite checkout at commit
`7b70111339b4af23cece95d63aeec1c705790868`. For each task,
`tools/tasktrove_runtime_probe.py`:

1. creates a fresh bounded shellsim environment;
2. mounts the task's `environment` source directory at the Dockerfile's last literal `WORKDIR`;
3. runs the unmodified `solution/solve.sh`;
4. mounts `/tests` and runs the unmodified `tests/test.sh` in the same environment; and
5. when verifier provisioning stops that wrapper, runs its Python tests directly as a fallback.

The probe records phase outcomes, resource stops, unsupported telemetry, command traces, stderr,
and verifier reward. It does not build the Dockerfile, install dependencies, start services, or
reproduce arbitrary wrapper setup. Results therefore answer “what support does this sample reach?”
rather than “how many TaskTrove tasks does shellsim solve?”

```sh
python3 tools/tasktrove_runtime_probe.py /path/to/OpenThoughts-TBLite > replay.json
```

## Baseline and cumulative revision result

All September 17, 2026 runs sample the same 100 tasks without a harness error. The final run adds
ordinary Python string and collection methods, VFS-backed pathlib operations, deterministic
Python and Bash random values, modeled `uv` environment and launcher options, and bounded pytest
CTRF output. It also adds ordinary Python syntax and argument binding, protocol-aware sequence and
set comparisons, a capability-free `statistics` module, and NumPy `@` plus multidimensional
strided indexing and assignment.

| Outcome | Golden before | Golden after | Verifier before | Verifier after | Wrapper before | Wrapper after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean exit | 47 | 49 | 13 | 18 | 13 | 16 |
| Explicit shellsim boundary | 38 | 36 | 39 | 44 | 80 | 79 |
| Other nonzero result | 12 | 12 | 44 | 33 | 3 | 1 |
| Resource exhaustion | 3 | 3 | 0 | 1 | 1 | 1 |
| Not run after exhaustion | 0 | 0 | 4 | 4 | 3 | 3 |

One task, `tsl-test-case-generation`, writes a positive partial reward of `0.8825`. This is useful
confirmation that solution state reaches a real verifier, but it is not a corpus pass-rate claim.
The task and reward are unchanged after the revision batch.

Five verifier payloads move from an explicit boundary or other failure to a clean exit:
`api-endpoint-permission-canonicalizer` after `str.isupper()`, `bash-log-processor-fix` after
`Path.relative_to()`, and `network-log-normalization` after `str.splitlines()`. The Bash `$RANDOM`
implementation moves `neural-architecture-search-final` from an AWK parse boundary to a clean
golden-solution exit. Protocol-aware tuple ordering moves `schedule-vacation` to a clean golden
exit. Adjacent string and f-string literals let `sympy-bug-fix` run its wrapper and verifier
cleanly, and `log-summary` also reaches a clean verifier.

Other additions expose the next honest boundary without changing the headline outcome. Collection
`copy()` and set ordering let `service-deployment-wave-planner` continue to a later module
boundary; `Path.stat()` and `json.JSONDecodeError` let `scan-linux-persistence-artifacts` run rather
than fail during parsing; and `random` lets the gRPC verifier continue to its missing `socket`
dependency. `statistics`, `Path.parts`, and `os.access` similarly move their tasks to later
`argparse` or invalid-program behavior. These are forward movement, not passes.

Verifier-wrapper ergonomics improve separately. Clean wrapper exits rise from 13 to 15, explicit
wrapper boundaries fall from 80 to 77, and one former boundary becomes an ordinary downstream
failure. The repeated `uv venv` boundary falls from 18 tasks to zero, `uv init` from four to zero,
the generic `uv` launcher-option boundary from 23 to zero, pytest `--ctrf` from two to zero, and
`uv pip install --system` from one to zero. Unbundled `--with` packages now retain specific package
boundaries such as pandas, requests, and httpx.

## Remaining reached boundaries

The broadest remaining accidental gap is still Python language and stdlib behavior. Reached
examples include assignment expressions in comprehensions, a few unsupported expression forms,
`argparse.add_subparsers`, `importlib.util`, `os.chdir`, and the exception family used by
`FileNotFoundError`. Each needs a coherent binding, loader, process-state, or exception contract;
the sample no longer identifies another safe syntax shortcut in the same class as this batch.

Missing third-party ecosystems are the next visible group, but most are intentional hard
boundaries. Pandas stops five solutions and four verifiers. Pydantic stops three verifiers.
Unbundled packages stop six solution tasks. Implementing partial pandas, databases, web stacks, or
native scientific packages would violate shellsim's boundary policy.

Small deterministic modules remain plausible additions. Verifiers reach `importlib` in two tasks
and reach `shutil`, `traceback`, and `zipfile` in one task each. `sqlite3` is the most frequent
missing verifier module at five tasks, but it should remain absent until shellsim can offer a
coherent database contract rather than a partial module. The 49 reached wrapper uses of `apt-get`
remain an explicit image-provisioning boundary.

Unavailable system binaries such as `openssl`, `debugfs`, `setfacl`, compilers, service tools, and
language runtimes each block only one or two golden solutions. The sample does not justify adding
partial implementations of them.
