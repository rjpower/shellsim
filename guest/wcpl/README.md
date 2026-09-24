# WCPL compiler fixture

`wcpl.wasm` is a checked-in compiler guest built from [WCPL] revision
`458a542ca81fa7a8fd8c8eed38a32dce8ed45135` under its MIT license. Its SHA-256 is
`f5983d50c424dd3bf737b81f1478df1a5f6d8598166de256fce9bc67f56d3f34`. The build embeds a
date string, so rebuilding on another day changes the hash without changing the compiler logic.

To regenerate it in a separate checkout of that revision:

```sh
cc -O2 -o wcpl-native w.c l.c p.c c.c
./wcpl-native -q -o wcpl.wasm w.c l.c p.c c.c
```

The integration test installs the fixture as `/usr/bin/wcpl` in the virtual filesystem. It
compiles a virtual C file into a virtual executable, and the shell executes that output. Neither
product code nor the test invokes a host compiler or reads host files at runtime. The fixture is
not installed in the default environment.

This proves the single-invocation compiler and linker path for self-contained one- and two-file
C programs. It does not establish general C or libc compatibility. The Wasm-hosted compiler fails to
resolve its embedded `<stdio.h>` because it constructs a backslash-separated resource path;
locally bypassing that lookup still produced incorrect or invalid binaries in the stdio cases
probed. Keep libc-backed builds outside the supported surface until that is understood.

[WCPL]: https://github.com/false-schemers/wcpl
