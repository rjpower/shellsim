"""Package-owned source for calling the host's virtual tool endpoint."""

import json
import subprocess


ENDPOINT = "http://host.shellsim/tools"


class ToolClient:
    def call(self, name, arguments):
        response = subprocess.check_output(
            [
                "curl", "-fsS", "-X", "POST", "-H", "Content-Type: application/json", "-d",
                json.dumps({"tool": name, "arguments": arguments}), ENDPOINT,
            ],
        )
        reply = json.loads(response)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]
