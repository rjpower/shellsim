"""Exercise real model/ASGI behavior without host network access."""

import json

from fastapi import FastAPI

app = FastAPI()


@app.get("/double/{n}")
async def double(n: int):
    return {"value": n * 2}


async def request(path):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("guest", 0),
            "server": ("guest", 80),
        },
        receive,
        send,
    )
    return messages


def complete(operation):
    try:
        operation.send(None)
    except StopIteration as result:
        return result.value
    operation.close()
    raise AssertionError("immediate ASGI request unexpectedly suspended")


valid = complete(request("/double/7"))
assert valid[0]["status"] == 200 and json.loads(valid[1]["body"]) == {"value": 14}
invalid = complete(request("/double/seven"))
assert invalid[0]["status"] == 422
assert app.openapi()["paths"]["/double/{n}"]["get"]["responses"]["200"]

print("fastapi: operations passed")
