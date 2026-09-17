# TaskTrove shell-command coverage

This note is a static prioritization sample for shellsim's command surface. It is not a task pass
rate and it does not claim that a registered command implements every option used by a task.

## Sample and method

The sample is the 100-task OpenThoughts-TBLite checkout at commit
`7b70111339b4af23cece95d63aeec1c705790868`. The checked-in
`tools/tasktrove_command_census.py` scans executable positions in `.sh` and `.bash` files,
classifies literal command names against shellsim's registry, and groups normalized argv forms for
commands marked partial. It also extracts coarse AWK, sed, and jq language features. Reproduce it
with:

```sh
python3 tools/tasktrove_command_census.py /path/to/OpenThoughts-TBLite > census.json
```

Schema version 2 reports invocation and task counts, option frequencies, embedded-language
features, normalized forms, dynamic executable positions, and a static assessment:

- `supported_surface`: every visible option and language feature is in a narrow profile backed by
  shellsim's implementation and compatibility tests.
- `explicit_boundary`: at least one visible option or feature is intentionally rejected.
- `dynamic`: shell expansion prevents static classification of the interface.
- `unclassified`: no narrow profile exists, so the tool makes no compatibility claim.

`supported_surface` is not a task pass result. It establishes that the visible interface is inside
the declared boundary, not that arbitrary input or hidden file state has the expected semantics.
The approximation can still overcount shell functions, generated scripts, and shell syntax. A
`missing` row remains a review queue, not proof of a missing binary.

## Result after the command-surface pass

| Source bucket | Calls | Real | Partial | Explicitly unsupported | Lexically unmatched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference solutions | 1,380 | 963 | 201 | 62 | 154 |
| Verifier scripts | 941 | 653 | 168 | 112 | 8 |
| Environment scripts | 348 | 191 | 58 | 17 | 82 |

The main actionable solution gap was `dd`: 27 calls in one repair task use byte-sized `seek`,
`count`, and `conv=notrunc`. Shellsim now supports that pattern, plus bounded `/dev/zero` creation.
The other repeated external programs have hard boundaries: `openssl` (20 solution calls),
`setfacl` (11), `nohup`, `debugfs`, `journalctl`, and service or administration tools. They now use
the common unsupported-command contract: status 127, one standard diagnostic, and structured
invocation telemetry.

The verifier sample is dominated by deliberate package-manager boundaries: 105 `apt-get` calls.
The previously unclassified `netstat`, `ss`, `redis-cli`, and `redis-server` calls are now explicit
unsupported invocations. The eight remaining lexical misses are local counters, generated scripts,
and brace/default-expansion artifacts, not general-purpose binaries.

The 154 unmatched solution tokens are likewise mostly task-local functions, numeric/data tokens,
and generated task executables. They should not be registered as global commands.

## Option-level result

| Source bucket | Partial calls | Supported surface | Explicit boundary | Dynamic surface | Unclassified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference solutions | 201 | 103 | 2 | 0 | 96 |
| Verifier scripts | 168 | 2 | 0 | 0 | 166 |
| Environment scripts | 58 | 37 | 0 | 0 | 21 |

For reference solutions, every observed `dd` (27 calls), jq (22), AWK (18), sed (8), and find (3)
form is now inside its declared surface. The census exposed one standard AWK `log()` use; shellsim
now implements it using the existing numeric evaluator. Grep has 25 supported-surface calls and two
explicit boundaries across two tasks: one uses unsupported PCRE mode (`-P`), and one supplies a
leading-dash certificate marker without `-e` or `--`, which a normal option parser treats as an
option.

The 96 unclassified solution calls are 83 Python entrypoints, nine pip entrypoints, two unzip calls,
and two ps calls. Python is intentionally not inferred from argv: a script path says nothing about
its syntax, imports, or library use. That surface needs source/import analysis or execution
telemetry rather than an optimistic CLI classification.

Verifier scripts are similarly dominated by program entrypoints rather than utility flags: 79
`uv`, 46 `pytest`, 25 `uvx`, 12 Python, three pip, and one timeout call remain unclassified. The
report retains their normalized forms and per-task counts so a later package-tool profile can be
added without rescanning the corpus.

There are also 47 dynamic executable positions in reference solutions and one in environment
scripts. They are reported separately from the 1,380 and 348 literal-command counts; all are
variable-selected commands and require runtime telemetry to resolve.
