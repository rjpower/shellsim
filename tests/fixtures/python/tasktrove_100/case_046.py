import json
value = {"z": 1, "a": 2}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
