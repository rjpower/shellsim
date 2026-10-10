"""Execute admitted PEP 517 hooks with pinned imports and target sysconfig.

The parent supplies a private source tree and a verified backend wheel closure.
Isolated host Python retains its admitted standard library; ambient packages,
PYTHONPATH and site startup hooks are excluded by the invocation and sys.path.
"""

import importlib
import json
import sys
from pathlib import Path


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    sys.path = (
        request["backend_paths"]
        + [request["imports"], request["configuration"]]
        + [path for path in sys.path if path and "site-packages" not in Path(path).parts]
    )
    from importlib.metadata import distributions

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {canonicalize_name(dist.metadata["Name"]): dist for dist in distributions(path=[request["imports"]])}
    checked = set()

    def require(raw: str) -> None:
        requirement = Requirement(raw)
        if requirement.url is not None:
            raise ValueError("direct-URL backend requirements are unsupported")
        if requirement.marker is not None and not requirement.marker.evaluate():
            return
        name = canonicalize_name(requirement.name)
        if name not in installed or installed[name].version not in requirement.specifier:
            raise ValueError("missing pinned backend requirement: " + raw)
        extras = requirement.extras or {""}
        for extra in extras:
            if (name, extra) in checked:
                continue
            checked.add((name, extra))
            for dependency in installed[name].requires or []:
                dependency = Requirement(dependency)
                if dependency.marker is None or dependency.marker.evaluate({"extra": extra}):
                    # The applicable extra marker was evaluated above.
                    dependency.marker = None
                    require(str(dependency))

    for requirement in request["requires"]:
        require(requirement)
    if list(sys.version_info[:2]) != request["python_version"]:
        raise ValueError("host and target Python major/minor versions differ")
    module, separator, attribute = request["backend"].partition(":")
    backend = importlib.import_module(module)
    if request["backend_paths"] and not any(
        Path(backend.__file__).resolve().is_relative_to(Path(path)) for path in request["backend_paths"]
    ):
        raise ValueError("in-tree backend was not loaded from its declared backend-path")
    if separator:
        for name in attribute.split("."):
            backend = getattr(backend, name)
    try:
        hook = backend.get_requires_for_build_wheel
    except AttributeError:
        additions = []
    else:
        additions = hook(request["config_settings"])
    if (
        not isinstance(additions, list)
        or len(additions) > 64
        or any(not isinstance(value, str) or len(value) > 4096 for value in additions)
    ):
        raise ValueError("invalid additional backend requirements")
    for requirement in additions:
        require(requirement)
    filename = backend.build_wheel(request["output"], request["config_settings"])
    Path(request["response"]).write_text(json.dumps({"wheel": filename, "additional_requires": additions}) + "\n")


if __name__ == "__main__":
    main()
