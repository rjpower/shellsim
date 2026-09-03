import json
print(json.dumps([1, {"x": "y"}], separators=(",", ":")))
