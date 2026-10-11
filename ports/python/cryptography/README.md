# Cryptography

Cryptography 50.0.2 builds its upstream locked Rust workspace against the
canonical PIC Rust stdlib, target CPython headers and static OpenSSL 3.5.9.
The Rust adapter translates Wasm waits for Shellsim's cooperative scheduler.
The wheel retains upstream Python sources, typing files and distribution
metadata. Its runtime dependencies include the existing native CFFI port.

The host CFFI generator installs into a private build directory with uv,
reusing the worker's normal download cache. Target C code uses the ordinary
cached WASI compiler wrapper. The generator does not enter the guest wheel.

The probe checks SHA-256, HMAC, HKDF, AES-GCM authenticated encryption and
Ed25519 signatures, including invalid tags and signatures. Socket binding
declarations omitted by OpenSSL's no-sock configuration are removed from
the upstream internal CFFI declarations. Threaded crypto, TLS transports,
dynamic OpenSSL providers and optional SSH/bcrypt support are unqualified.
