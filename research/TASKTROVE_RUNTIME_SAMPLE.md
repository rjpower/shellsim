# TaskTrove workflow replay

This note measures unchanged TaskTrove golden solutions and verifier payloads in shellsim. It is
the primary compatibility signal for the corpus. The static command census remains useful for
finding code that execution did not reach, but it is not a task pass-rate proxy.

## Sample and method

The sample is the 100-task OpenThoughts-TBLite checkout at commit
`7b70111339b4af23cece95d63aeec1c705790868`. `tools/tasktrove_runtime_probe.py` creates one fresh,
bounded shellsim environment per task and retains that environment across setup, the golden
solution, and the verifier.

The replay models the task image as shellsim's built-in userspace plus local Dockerfile `COPY`,
`ENV`, `WORKDIR`, and deterministic `RUN` steps. Network, operating-system, and package
provisioning steps are recorded as image prerequisites rather than executed. It then mounts and
runs the unmodified `solution/solve.sh`. Finally, it mounts `/tests` and runs all `test_*.py` files
with shellsim's pytest entry point, or the task's direct reference evaluator when it has no pytest
file. This bypasses only verifier-wrapper provisioning and report-upload boilerplate such as
`apt-get`, `curl`, and `uvx` installation.

Every phase records its status, resource stop, command trace, unsupported telemetry, and bounded
stderr. The final result also records verifier reward, skipped image prerequisites, and harness
errors. Reproduce it with an installed local Python build:

```sh
python3 tools/tasktrove_runtime_probe.py /path/to/OpenThoughts-TBLite > replay.json
```

## Result

The September 17, 2026 replay completes all 100 tasks without a harness error.

| Phase outcome | Golden solutions | Verifiers |
| --- | ---: | ---: |
| Clean exit | 50 | 14 |
| Explicit shellsim boundary | 40 | 40 |
| Nonzero without unsupported telemetry | 7 | 42 |
| Resource exhaustion | 3 | 1 |
| Not run after terminal exhaustion | 0 | 3 |

One task, `tsl-test-case-generation`, produces a positive partial reward of `0.8825`; no task
produces full reward. At the task level, 67 replays end at an explicit boundary, 28 are rejected by
the verifier, four exhaust modeled resources, and one produces the partial reward. Three of the 28
verifier-rejected tasks also record an earlier explicit boundary.

Runtime evidence resolves dynamic commands and exposes interactions that argv inspection cannot.
For example, one solution reaches AWK 1,010 times with a malformed expression because shellsim
does not model Bash's special `$RANDOM` variable; the static census saw valid AWK remainder syntax.
Other solution frontiers include Python language and API gaps in 11 tasks, unavailable pip
distributions in six, pandas in five, and explicit system-tool boundaries such as `apt-get`,
`openssl`, `debugfs`, `setfacl`, and service or compiler commands. Verifier boundaries are led by
Python language and API gaps in 12 tasks and missing modules such as `sqlite3`, pandas, pydantic,
and importlib.

## Interpretation and limitations

The verifier is the strongest available oracle, so a reward or assertion result takes precedence
over a static classification. The raw phase evidence remains more important than the aggregate
category. A verifier failure without unsupported telemetry can mean a shellsim semantic mismatch,
an earlier unmodeled image prerequisite, or a verifier assumption outside the replay mapping.

Eighty-eight tasks declare at least one image prerequisite that this replay does not provision.
Arbitrary Docker builds, downloaded native packages, services, compilers, databases, and non-root
user identity remain outside the model. The JSON retains those prerequisites instead of silently
claiming Docker equivalence. No task code gains host filesystem, process, network, environment, or
clock access.

The next compatibility work should follow repeated runtime evidence. Small deterministic Python
stdlib and API gaps, broader pytest collection, and deterministic Bash `$RANDOM` are plausible
targets. Native ecosystems and service workloads should remain explicit boundaries unless a
separate design gives them a complete, intelligible contract.
