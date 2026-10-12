"""Resolve upstream notifications by project and version without a network."""

from farama_notifications import notifications

affected = notifications["gymnasium_robotics"]
assert "reward functions were updated" in affected["1.2.1"]
assert affected["1.2.1"] == affected["1.4.2"]
assert affected.get("1.4.3") is None
assert notifications.get("shellsim-unknown-project") is None

print("farama-notifications: version lookup passed")
