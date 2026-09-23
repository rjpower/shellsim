# POSIX-derived shell corpus

These cases are small, independently written probes derived from the observable requirements in
POSIX.1-2024, Shell Command Language, sections 2.2, 2.5, 2.6, 2.7, and 2.9. They are not a
certification suite. They complement the pinned Oils cases with ordinary behavior that application
scripts depend on:

| Area | Cases |
| --- | --- |
| quoting, field splitting, positional parameters | `quoted-ifs`, `ifs-boundaries`, `star-at-expansion`, `positional-function` |
| parameter and command substitution | `parameter-defaults`, `parameter-trimming`, `parameter-error`, `subshell-command-sub` |
| redirection and virtual filesystem data flow | `ordinary-redirection`, `descriptor-ordering`, `heredoc-expansion`, `group-redirection` |
| POSIX loops, `case`, functions, and subshell scope | `control-flow`, `case-patterns`, `function-return-scope`, `subshell-directory-scope` |
| pipeline, negation, AND-OR, assignment, and export status | `status-lists`, `errexit-contexts`, `errexit-positive`, `errexit-or-positive`, `assignment-export`, `pipeline-scope` |
| stateful builtins and signals | `read-raw`, `read-escapes`, `getopts-options`, `trap-signal` |

Expected output was checked with Bash and dash in POSIX-compatible syntax. Cases normalize behavior
that POSIX permits shells to report with different nonzero statuses. Each file is hashed in the
manifest so a changed probe requires an explicit expectation review.
