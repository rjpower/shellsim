# HTTPX

Stages the official wheel and its existing dependency recipes. The guest probes
use upstream `MockTransport` for redirects, cookies, JSON bodies and HTTP errors,
and `ASGITransport` with the existing FastAPI and sniffio check dependencies.
Immediate coroutines avoid a socket-backed event loop; serving and suspended
operations are outside these probes.

HTTPcore declares the CPython SSL provider. Wheels retain upstream metadata.
