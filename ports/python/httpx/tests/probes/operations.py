"""Exercise upstream HTTPX request handling with its in-memory transport."""

import json

import httpx


def respond(request):
    if request.url.path == "/start":
        assert request.headers["authorization"] == "Basic dXNlcjpwYXNz"
        return httpx.Response(302, headers={"location": "/result?count=2", "set-cookie": "session=guest; Path=/"})
    if request.url.path == "/result":
        assert request.url.params["count"] == "2"
        assert request.headers["cookie"] == "session=guest"
        return httpx.Response(200, json={"path": request.url.path})
    if request.url.path == "/echo":
        return httpx.Response(201, json=json.loads(request.content))
    return httpx.Response(404)


with httpx.Client(
    transport=httpx.MockTransport(respond), base_url="http://guest.test", auth=("user", "pass"), trust_env=False
) as client:
    response = client.get("/start", follow_redirects=True)
    assert response.json() == {"path": "/result"}
    assert [item.status_code for item in response.history] == [302]
    echoed = client.post("/echo", json={"items": [1, 2], "label": "guest"})
    assert echoed.status_code == 201 and echoed.json() == {"items": [1, 2], "label": "guest"}
    try:
        client.get("/missing").raise_for_status()
    except httpx.HTTPStatusError as error:
        assert error.response.status_code == 404
    else:
        raise AssertionError("HTTP error status was accepted")

print("httpx: mock operations passed")
