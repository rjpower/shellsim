# FastAPI

The official wheel uses the declared Starlette and native Pydantic closure.
The guest probe calls the real ASGI application with supplied receive/send
callbacks, covering integer route parameters, JSON responses, invalid-input
responses and OpenAPI schema generation. These immediate coroutines finish
without an event loop or network listener.

The ordinary asyncio event loop needs a socket pair for wakeups, which the
current CPython WASI runtime does not provide. Network serving, suspended ASGI
transport operations and synchronous routes requiring a thread pool are not
qualified by this port. No host socket or loop is substituted.
