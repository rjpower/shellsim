# Shellsim POSIX interfaces

This port builds the shared libc bridge used by guest build tools. Its static
archive supplies virtual process creation, wait/exec, descriptor operations,
working directories, temporary files and bounded account lookup. The source
continues to live beside its owning platform modules under `ports/toolchain`.
The recipe pins every source/header by hash and stages only those files for
an ordinary CMake build.

Consumers declare `native/shellsim-posix` as a target dependency. The graph
supplies its headers and archive through the dependency prefix. The installed
`shellsim-posix.pc` declares the required signal/open linker wrappers and
header declarations, including the process ABI's nonzero `O_CLOEXEC` flag.
Consumers using configure may need to force these declarations only during
compilation, since configure also preprocesses empty files to detect flags.

The current recipe targets the threaded v3 WASI cohort. Linking this archive
into a different target or runtime profile requires a separately built artifact.
Calls remain inside Shellsim's virtual kernel; no guest operation reaches host
processes or descriptors. Unsupported signal callbacks and process attributes
retain the existing explicit errors from the platform implementation.

The port-local native check exercises descriptor I/O and invalid descriptors,
exclusive temporary files, and a spawned shell child's exit status. It is run
through the graph release and public environment setup path with `--check`.
