"""Exercise local public behavior in the actual guest."""

import anyio
from anyio import WouldBlock

send, receive = anyio.create_memory_object_stream(1)
send.send_nowait(7)
assert receive.receive_nowait() == 7
try:
    receive.receive_nowait()
except WouldBlock:
    pass
else:
    raise AssertionError("empty stream did not block")
send.close()
receive.close()
print("anyio: guest probe passed")
