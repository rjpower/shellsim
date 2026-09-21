# Shellsim repository guidance

These instructions apply to the entire repository. They adapt Marin's contribution discipline to
shellsim's Rust codebase. See `CONTRIBUTING.md` for the human-facing workflow.

## Before changing code

- Read the nearest module documentation and tests before editing.
- Keep changes tied to a concrete behavior, bug, or documented design goal. Avoid speculative
  refactors and unrelated cleanup.
- Preserve existing user changes in a dirty worktree.
- Treat simulation boundaries as security boundaries: simulated programs must not gain ambient
  host filesystem, process, network, environment, or clock capabilities.

## Code guidelines

- Give each production Rust module a `//!` overview describing its purpose and important design
  constraints. Add a test-file strategy comment when the strategy is not obvious.
- Document public APIs and non-trivial functions with context, invariants, inputs, outputs, and an
  example when it clarifies behavior. Do not paraphrase the implementation.
- Prefer typed structs and enums over loosely structured strings or maps.
- Keep functions and modules focused. Extract reusable pieces after repeated use or when doing so
  makes a security or correctness boundary explicit; otherwise follow the rule of three.
- Use checked or saturating arithmetic at untrusted-data and resource-accounting boundaries.
- Reject unsupported syntax and capabilities explicitly. Never fall back to host execution.
- Meter loops, allocation, output, filesystem growth, and other work reachable from simulated
  input before performing unbounded host work.
- Keep resource costs simple and roughly proportional to real work. Order-of-magnitude CPU and
  memory estimates are sufficient; prefer a conservative constant or linear bound over complex
  exact accounting. Resource limits are safety and scheduling controls, not a profiler.
- Add a small unit test for tricky logic and an integration or differential test for observable
  compatibility behavior.

## Validation workflow

Run a narrow test while iterating, for example:

```sh
cargo test --test time_virtualization
cargo test python::parser
```

Before handing off a code change, run the same repository gates used by CI:

```sh
./infra/pre-commit.py --all-files
./infra/ci/run_tests.py
```

Use the toolchain pinned in `rust-toolchain.toml`; do not silently update it as part of an
unrelated change.

Use `./infra/pre-commit.py --all-files --fix` to apply formatting before linting. Do not weaken a
lint, skip a test, or update a differential fixture merely to make a gate green; explain and test
the intended behavior.

## Tests and reviews

- Tests must be deterministic and must not rely on elapsed host time, host locale, network access,
  or unordered collection output.
- Prefer semantic assertions over snapshots. Use checked output fixtures when exact compatibility
  is the contract and record their provenance.
- Differential tests may invoke a reference implementation only from the test harness. Product
  code must remain capability-free.
- Cover success, invalid input, resource exhaustion, and the explicit unsupported frontier when
  adding a parser, VM, command, or stdlib feature.

## Writing and pull requests

- Use plain, sober technical English. Lead with the result, decision, or changed behavior and
  support claims with named evidence.
- State scope, uncertainty, and limitations directly. Avoid hype, generic claims of importance,
  filler, decorative headings, emoji, stock contrasts, synonym rotation, and em-dash asides.
- Write imperative pull-request and commit titles of at most 72 characters. An optional `[scope]`
  prefix is fine; do not use Conventional Commit prefixes such as `feat:` or `fix:`.
- In a handoff or pull request, describe behavior before motivation, then give validation evidence
  and material caveats. Do not narrate the diff file by file or add an empty `Testing` scaffold.
- Make the description stand on its own as a useful squash-commit message. Link the originating
  issue with `Fixes #NNN` or `Part of #NNN` when relevant.
