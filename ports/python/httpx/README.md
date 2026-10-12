# HTTPX

Stages the official wheel and its existing dependency recipes. The guest probes
use upstream `MockTransport` for redirects, cookies, JSON bodies and HTTP errors,
and `ASGITransport` with the existing FastAPI and sniffio check dependencies.
Immediate coroutines avoid a socket-backed event loop; serving and suspended
operations are outside these probes.

HTTPcore's standalone guest probe requires a CPython SSL provider, which this
checkout does not supply. HTTPX's mock probe passes with its declared closure.
The wheels are unchanged.
