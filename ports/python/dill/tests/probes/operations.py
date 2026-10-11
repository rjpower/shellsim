"""Exercise a bounded public API inside the actual guest."""

import dill

value = dill.loads(dill.dumps(lambda x: x + 2))
assert value(3) == 5
print("dill: public API probe passed")
