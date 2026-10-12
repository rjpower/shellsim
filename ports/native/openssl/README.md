# OpenSSL

OpenSSL 3.5.9 builds upstream static PIC archives with the selected WASI C
compiler, sccache and the graph's requested job count. The dependency graph
includes Shellsim's existing POSIX archive for virtual process identity.
The native probe checks SHA-256 through both the direct and EVP interfaces.

This configuration disables assembly, sockets, dynamic providers, engines,
async operations, console prompts and mmap-backed secure memory. Thread support
uses the guest pthread runtime, as required by CPython's `_ssl` extension.
Entropy uses WASI getentropy, which remains within the simulator's random
source. The archives support in-process crypto operations; they do not
provide host networking, host devices or protected host memory.
