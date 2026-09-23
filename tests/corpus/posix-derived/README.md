# POSIX-derived shell corpus

These cases are small, independently written probes derived from the observable requirements in
POSIX.1-2024, Shell Command Language, sections 2.2, 2.5, 2.6, 2.7, and 2.9. They are not a
certification suite. They complement the pinned Oils cases with ordinary behavior that application
scripts depend on:

| Area | Cases |
| --- | --- |
| quoting, field splitting, positional parameters | `quoted-ifs`, `positional-function` |
| parameter and command substitution | `parameter-defaults`, `subshell-command-sub` |
| redirection and virtual filesystem data flow | `ordinary-redirection` |
| POSIX loops, `case`, functions, and subshell scope | `control-flow`, `positional-function`, `subshell-command-sub` |
| pipeline, negation, AND-OR, assignment, and export status | `status-lists`, `assignment-export` |

Expected output was checked with Bash 5.2 in POSIX-compatible syntax. Each file is hashed in the
manifest so a changed probe requires an explicit expectation review.
