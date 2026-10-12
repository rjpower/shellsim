"""Parse HTTP/1.1 responses through upstream HTTPcore's mock backend."""

import httpcore

backend = httpcore.MockBackend(
    [b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"3\r\nabc\r\n", b"2\r\nde\r\n0\r\n\r\n"]
)
with httpcore.ConnectionPool(network_backend=backend) as pool:
    response = pool.request("GET", "http://guest.test/resource")
    assert response.status == 200 and response.content == b"abcde"
    assert response.extensions["http_version"] == b"HTTP/1.1"

with httpcore.ConnectionPool(
    network_backend=httpcore.MockBackend([b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nx"])
) as pool:
    try:
        pool.request("GET", "http://guest.test/truncated")
    except httpcore.RemoteProtocolError:
        pass
    else:
        raise AssertionError("truncated response was accepted")

print("httpcore: mock protocol operations passed")
