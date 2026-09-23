# Compatibility corpus

Shellsim measures compatibility with frozen programs running in a deterministic virtual
environment. This complements focused unit and differential tests. It does not claim POSIX
certification or general host compatibility.

## Stock agent profile

`stock_agent_v1` provides `/home/shellsim`, `/work`, `/tmp`, `/bin`, `/usr/bin`, and small virtual
`/etc/passwd` and `/etc/group` files. Programs start in `/work` as uid 1000 with conventional
`HOME`, `USER`, `LOGNAME`, `PATH`, `TMPDIR`, locale, and UTC variables. The virtual clock,
filesystem, process table, and network retain the same deterministic boundaries as other shellsim
entrypoints. The profile name is versioned because changing these inputs can change workload
behavior.

Run a checked corpus with:

```sh
shellsim corpus tests/corpus/stock-smoke/manifest.json > report.json
```

Each case receives a fresh environment. Fixture sources are plain paths relative to the manifest;
destinations are absolute VFS paths. Optional SHA-256 values pin fixture bytes. The runner rejects
source traversal and symlink escape, bounds manifests, fixtures, case counts, and execution
resources, and never exposes a host path to simulated code.

Cases declare one disposition:

- `pass`: status and any checked output must match;
- `frontier`: the workload is executed and must fail through a classified boundary;
- `skip`: execution is intentionally omitted and a reason is required.

Reports distinguish setup, parsing, unknown-command, unsupported-feature, missing-module,
fixture, semantic, runtime, resource, hang, and capability failures. Some classes are preventive
schema categories and should remain empty. A known frontier is not counted as a pass, even when it
is expected.

Frontier cases must set `class` to require one exact failure category. This prevents a known missing
dependency from silently turning into a parser failure, hang, or resource regression.

Compact derived cases may put shell or Python source in `code`, UTF-8 standard input in `stdin`,
and UTF-8 expectations in `stdout` and `stderr`. Inline fixtures use `contents`. Imported fixtures
continue to use pinned relative `source` paths and optional hashes. These forms are mutually
exclusive, and ambiguous manifests are rejected before any case runs.

Cases that generate files can assert exact UTF-8 contents with `expect.files`, keyed by absolute
virtual paths. The runner bounds each read by the expected length before comparing bytes.

Cases attach stable capability identifiers through `covers`. Reports aggregate passing, frontier,
skipped, and unexpectedly failing observations for each identifier. Frontier cases must declare
an exact `class`; a boundary changing from an unsupported feature to a parse error is a regression.

`pass` cases may declare exact `unsupported` and `unsupported_commands` lists. This is useful when
the behavior under test intentionally invokes a missing command, such as a parser ambiguity whose
contract includes status 127. Undeclared entries remain failures.

## Corpus policy

Every imported program must record its source URL, pinned revision, path, content hash, and
license in the corpus README or manifest-adjacent provenance file. Reference interpreters run
only in trusted test tooling. Checked outputs let ordinary CI remain hermetic.

Pull requests run a small reviewed gold corpus. Larger frozen corpora may run in nightly jobs.
Live source discovery must remain separate from gating CI. Promote candidates only after their
fixtures, dependencies, license, and expected behavior have been reviewed.

The current Python baseline remains the 100-case TaskTrove differential corpus under
`tests/python/corpus/tasktrove`. All 100 cases are expected to match checked CPython 3.14 output.
The compatibility runner adds reviewed suites without replacing that unchanged-source test:

- `micropython-basics` freezes 100 unchanged language tests with checked CPython output;
- `python-derived` supplies small readable contracts for language composition and commonly used
  APIs, including exact frontiers for support work;
- `oils-spec` freezes 50 shell-specification case bodies with upstream provenance;
- `posix-derived` supplies compact standards-oriented probes for common behavior underrepresented
  in the imported suite.
- `whole-programs` runs maintained shell probes and configure paths while keeping broad ShellSpec
  and Modernish composition frontiers explicit.

These suites measure compatibility breadth. They do not confer POSIX certification or complete
CPython compatibility.
