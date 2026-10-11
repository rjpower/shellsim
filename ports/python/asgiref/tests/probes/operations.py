"""Exercise local public behavior in the actual guest."""

from asgiref.local import Local

local = Local()
local.answer = 42
assert local.answer == 42
del local.answer
try:
    value = local.answer
except AttributeError:
    pass
else:
    raise AssertionError("deleted local value survived")
print("asgiref: guest probe passed")
