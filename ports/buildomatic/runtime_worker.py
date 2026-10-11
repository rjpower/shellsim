"""Publish a cached interpreter without loading compiler or resolver products."""

import json
import os
from pathlib import Path

from ports.buildomatic.artifacts import install, register, resolve, write_reference
from ports.buildomatic.delivery import DeliveryStore, publish_runtime
from ports.buildomatic.remote_store import RemoteStore


def main() -> None:
    specification = json.loads((Path(os.environ["BUILD_INPUT_recipe"]) / "node.json").read_text())
    store = RemoteStore(specification["prefix"])
    component = specification["component"]
    record = resolve(store, "runtime/" + component["receipt_id"])
    if record is None:
        bundle = install(store, component["artifact"], Path("/app/.buildomatic/artifacts"))
        runtime = publish_runtime(bundle, component["receipt_id"], DeliveryStore(specification["prefix"]))
        record = {"runtime": runtime}
    register(store, "runtime/" + component["receipt_id"], record)
    register(store, "runtime/" + specification["sdk_name"], record)
    write_reference(Path(os.environ["BUILD_OUTPUT_DIR"]) / "result.json", record)


if __name__ == "__main__":
    main()
