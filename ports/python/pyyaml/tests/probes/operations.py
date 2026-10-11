"""Use the upstream pure YAML parser and reject unsafe or malformed input."""

import yaml

assert not yaml.__with_libyaml__
data = {"name": "ports", "values": [1, 2], "enabled": True}
assert yaml.safe_load(yaml.safe_dump(data)) == data
for source in ("items: [", "!!python/object:builtins.object {}"):
    try:
        yaml.safe_load(source)
    except yaml.YAMLError:
        pass
    else:
        raise AssertionError("invalid or unsafe YAML accepted")
print("pyyaml: pure parser, round-trip and rejection passed")
