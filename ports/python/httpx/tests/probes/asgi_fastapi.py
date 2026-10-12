"""Run in the integrated guest with the existing FastAPI and sniffio ports."""

import httpx
import sniffio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Payload(BaseModel):
    value: int


@app.post("/double/{factor}")
async def double(factor: int, payload: Payload):
    return {"value": factor * payload.value}


@app.get("/failure")
async def failure():
    raise ValueError("application failure")


async def exercise():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://guest.test", trust_env=False
    ) as client:
        response = await client.post("/double/3", json={"value": 7})
        assert response.status_code == 200 and response.json() == {"value": 21}
        invalid = await client.post("/double/three", json={"value": "bad"})
        assert invalid.status_code == 422
        assert {item["type"] for item in invalid.json()["detail"]} == {"int_parsing"}
        assert len(invalid.json()["detail"]) == 2
        schema = await client.get("/openapi.json")
        assert "Payload" in schema.json()["components"]["schemas"]
        try:
            await client.get("/failure")
        except ValueError:
            pass
        else:
            raise AssertionError("ASGI application error was swallowed")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://guest.test",
        trust_env=False,
    ) as client:
        response = await client.get("/failure")
        assert response.status_code == 500


# Immediate ASGI coroutines need no socket-backed event loop in the WASI guest.
token = sniffio.current_async_library_cvar.set("asyncio")
operation = exercise()
try:
    try:
        operation.send(None)
    except StopIteration:
        pass
    else:
        raise AssertionError("ASGI operation unexpectedly suspended")
finally:
    operation.close()
    sniffio.current_async_library_cvar.reset(token)

print("httpx: FastAPI ASGI operations passed")
