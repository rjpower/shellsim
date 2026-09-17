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

## Result

The September 17, 2026 run samples all 100 tasks without a harness error.

| Outcome | Golden solution | Selected verifier |
| --- | ---: | ---: |
| Clean exit | 47 | 13 |
| Explicit shellsim boundary | 38 | 39 |
| Other nonzero result | 12 | 44 |
| Resource exhaustion | 3 | 1 |
| Not run after exhaustion | 0 | 4 |

One task, `tsl-test-case-generation`, writes a positive partial reward of `0.8825`. This is useful
confirmation that solution state reaches a real verifier, but it is not a corpus pass-rate claim.

## Where support would help

The broadest reached gap is Python itself. Language or ordinary object/API behavior stops 13
solutions and 11 verifier payloads. The raw report preserves the exact diagnostic for each task;
examples include unsupported expression and parameter forms, `argparse.add_subparsers`, string and
path methods, ordering, and regex behavior. These are better candidates than adding another
one-off binary because each extends the common execution substrate.

Missing third-party ecosystems are the next visible group, but most are intentional hard
boundaries. Pandas stops five solutions and four verifiers. Pydantic stops three verifiers.
Unbundled packages stop six solution tasks. Implementing partial pandas, databases, web stacks, or
native scientific packages would violate shellsim's boundary policy.

Small deterministic modules remain plausible additions. Verifiers reach `importlib` in two tasks
and reach `random`, `shutil`, `traceback`, and `zipfile` in one task each. `sqlite3` is the most
frequent missing verifier module at five tasks, but it should remain absent until shellsim can
offer a coherent database contract rather than a partial module.

The shell sample exposes one ordinary Bash omission: `$RANDOM` is empty. One architecture-search
solution consequently sends malformed remainder expressions to AWK 1,010 times. A deterministic
modeled `$RANDOM` would be small and broadly intelligible even though it occurs in only one sampled
task.

Verifier wrappers also show low-cost CLI compatibility opportunities: `uv` launcher options occur
in 23 tasks, `uv venv` arguments in 18, and pytest's `--ctrf` in two. These are harness ergonomics,
not task functionality, and should be accepted only when shellsim can give them harmless,
predictable semantics. The 49 wrapper uses of `apt-get` remain an explicit image-provisioning
boundary.

Unavailable system binaries such as `openssl`, `debugfs`, `setfacl`, compilers, service tools, and
language runtimes each block only one or two golden solutions. The sample does not justify adding
partial implementations of them.
