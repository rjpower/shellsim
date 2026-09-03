# TaskTrove Python differential corpus

This directory contains 100 deterministic standalone Python mini-scripts derived from syntax and
stdlib idioms observed in the locally mirrored `open-thoughts/OpenThoughts-TBLite` corpus at
`/tmp/openthoughts-tblite-7b70111`. The manifest records originating task and source path. Scripts
do not access host files, network, clock, randomness, or processes.

Each fixture is a minimized behavioral probe: it preserves a Python syntax/API idiom seen at the
manifested source location, but does not copy the task solution or claim to reproduce its whole
behavior. `provenance.tsv` records the source symbol or line and the derivation decision for every
case. Some source paths are shell wrappers whose embedded heredoc is Python; line numbers refer to
the wrapper exactly as extracted. Cases are allowed to use a related task when that is where the
idiom is actually present. No fixture is labeled as corpus-derived solely because its path exists.

`supported` cases are the compatibility contract: shellsim output/status/stderr must match the
checked `.out` file and CPython 3.14 when available. `frontier` cases are valid CPython programs
outside the current slice; their CPython output remains checked, while shellsim must reject them.
Promote a frontier case only after reviewing the emulator and its differential result.
