"""Package-owned source for calling the host's virtual tool endpoint."""

import json
from urllib.request import Request, urlopen


ENDPOINT = "http://host.shellsim/tools"


class ToolClient:
    def call(self, name, arguments):
        request = Request(
            ENDPOINT,
            data=json.dumps({"tool": name, "arguments": arguments}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            reply = json.loads(response.read().decode("utf-8"))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]
