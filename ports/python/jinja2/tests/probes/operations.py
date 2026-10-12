"""Exercise local public behavior in the actual guest."""

from jinja2 import Environment, StrictUndefined, UndefinedError

env = Environment(autoescape=True, undefined=StrictUndefined)
assert env.from_string("{{ name }}:{{ values|sum }}").render(name="<x>", values=[1, 2]) == "&lt;x&gt;:3"
try:
    env.from_string("{{ missing }}").render()
except UndefinedError:
    pass
else:
    raise AssertionError("undefined value accepted")
print("jinja2: guest probe passed")
