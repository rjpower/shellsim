# Oils shell semantics sample

This corpus contains exact test bodies extracted from the Oils shell specification suite at commit
`08310e96b182cf1d2dd65161e07b1743d17f8936`. The upstream project is Apache-2.0 licensed; the
license text is included beside this file. Each case records its upstream path, case number, commit,
and content hash in `manifest.json`.

The sample covers assignment, command substitution, exit status, loops, globbing, redirection,
word splitting, and parameter tests. Cases were retained only when their checked upstream behavior
also agreed with Bash. Some are deliberate Bash extensions and therefore complement, rather than
replace, the POSIX-derived corpus.
