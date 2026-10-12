"""Typed node requests shared by graph planning and local or pooled execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from ports._support.graph import Dependency, Port
from ports.buildomatic.cache import record_key


class Operation(str, Enum):
    BUILD = "build"
    DELIVERY = "delivery"
    RUNTIME = "runtime"
    PROBE = "probe"
    CHECK = "check"


@dataclass(frozen=True)
class Input:
    name: str
    reference: str
    key: str


@dataclass(frozen=True)
class Node:
    reference: str
    operation: Operation
    key: str
    port: Port
    inputs: tuple[Input, ...] = ()
    producer: dict | None = None
    test: dict | None = None

    @property
    def record_key(self) -> str:
        return record_key(self.key)

    @property
    def action_id(self) -> str:
        return self.operation.value + "-" + self.key[:24]

    def specification(self, prefix: str, jobs: int, offline: bool) -> dict:
        return {
            "reference": self.reference,
            "operation": self.operation.value,
            "key": self.key,
            "port": {
                "reference": self.port.reference,
                "name": self.port.name,
                "version": self.port.version,
                "digest": self.port.digest,
                "dependencies": [asdict(edge) for edge in self.port.dependencies],
                "recipe": self.port.recipe,
                "variant": self.port.variant,
            },
            "inputs": [asdict(item) for item in self.inputs],
            "producer": self.producer,
            "test": self.test,
            "prefix": prefix,
            "jobs": jobs,
            "offline": offline,
        }


def decode(value: dict, ports: Path) -> Node:
    port = value["port"]
    return Node(
        value["reference"],
        Operation(value["operation"]),
        value["key"],
        Port(
            port["reference"],
            ports / Path(port["reference"].partition(":")[0]).parent,
            port["name"],
            port["version"],
            port["digest"],
            tuple(Dependency(**edge) for edge in port["dependencies"]),
            port["recipe"],
            variant=port["variant"],
        ),
        tuple(Input(**item) for item in value["inputs"]),
        value["producer"],
        value["test"],
    )
