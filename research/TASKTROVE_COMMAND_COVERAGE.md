# TaskTrove shell-command coverage

This note is a static prioritization sample for shellsim's command surface. It is not a task pass
rate and it does not claim that a registered command implements every option used by a task.

## Sample and method

The sample is the 100-task OpenThoughts-TBLite checkout at commit
`7b70111339b4af23cece95d63aeec1c705790868`. The checked-in
`tools/tasktrove_command_census.py` scans executable positions in `.sh` and `.bash` files and
classifies literal command names against shellsim's registry. Reproduce it with:

```sh
python3 tools/tasktrove_command_census.py /path/to/OpenThoughts-TBLite > census.json
```

This approximation overcounts shell functions, dynamic expansions, generated scripts, and some
shell syntax. In particular, a `missing` row is a review queue, not proof of a missing binary.

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
and generated task executables. They should not be registered as global commands. The next useful
coverage work is option-level compatibility measurement for commands already marked partial,
especially Python, `grep`, `jq`, `awk`, and `sed`, rather than manufacturing stubs for those names.
