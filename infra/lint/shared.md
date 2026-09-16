# Shellsim lint-review protocol

This catalog is an advisory review layer for shellsim. It complements rustfmt, Clippy, Ruff, and
tests. It does not report formatting, import order, compiler-detectable mistakes, or speculative
improvements without a concrete defect in the changed code.

## Inputs

Review every changed file against the supplied merge base. Inspect each named file with read-only
Git commands and read enough surrounding code to judge the change. Follow a call into unchanged
code only when a rule requires confirmation. Skip lock files, generated output, binary data,
checked compatibility fixtures, and vendored files.

Report only concerns introduced or exposed by the change. Simulation boundaries are security
boundaries: product code must not acquire ambient host filesystem, process, network, environment,
or clock access. Pay particular attention to work performed before resource accounting.

## Confidence and suppression

Emit a finding only at confidence 0.70 or higher. Prefer silence to a weak inference. A nearby
comment of the form `lint-review: allow <code>` suppresses that rule when it explains why the
exception is safe.

When two rules describe the same issue, emit the more specific rule. Different defects on the
same line remain separate findings. The meta lane owns whole-change shapes and must not repeat a
local finding.

## Output format

Emit exactly one finding per line:

```
<path>:<line>: <code> (<confidence>) <message>
```

Use repository-relative paths, post-change 1-indexed line numbers, two decimal confidence, and a
message of at most 200 characters. Emit no preamble, summary, Markdown fence, or "no findings"
message. Empty output is correct.
