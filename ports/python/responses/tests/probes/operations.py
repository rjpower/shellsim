"""Exercise local public behavior in the actual guest."""

import requests
import responses

with responses.RequestsMock() as mock:
    mock.get("http://example.test/api", json={"value": 7})
    assert requests.get("http://example.test/api").json() == {"value": 7}
    assert len(mock.calls) == 1
print("responses: guest probe passed")
