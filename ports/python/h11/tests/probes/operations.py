"""Exercise incremental HTTP framing and malformed-body rejection."""

import h11

client = h11.Connection(h11.CLIENT)
server = h11.Connection(h11.SERVER)
wire = client.send(h11.Request(method="POST", target="/echo", headers=[("Host", "guest"), ("Content-Length", "3")]))
wire += client.send(h11.Data(data=b"abc")) + client.send(h11.EndOfMessage())
server.receive_data(wire[:12])
assert server.next_event() is h11.NEED_DATA
server.receive_data(wire[12:])
request = server.next_event()
assert request.method == b"POST" and request.target == b"/echo"
assert server.next_event().data == b"abc"
assert isinstance(server.next_event(), h11.EndOfMessage)
response = server.send(h11.Response(status_code=200, headers=[("Content-Length", "3")]))
response += server.send(h11.Data(data=b"abc")) + server.send(h11.EndOfMessage())
client.receive_data(response)
assert client.next_event().status_code == 200
assert client.next_event().data == b"abc"
assert isinstance(client.next_event(), h11.EndOfMessage)

broken = h11.Connection(h11.SERVER)
broken.receive_data(b"POST / HTTP/1.1\r\nHost: guest\r\nContent-Length: 5\r\n\r\nx")
broken.next_event()
broken.next_event()
broken.receive_data(b"")
try:
    broken.next_event()
except h11.RemoteProtocolError:
    pass
else:
    raise AssertionError("incomplete body was accepted")

print("h11: protocol operations passed")
