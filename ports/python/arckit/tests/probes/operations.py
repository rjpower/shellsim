"""Exercise task grid conversion and scoring without external ARC data."""

from arckit import Task, fmt_grid

examples = [{"input": [[0, 1]], "output": [[1, 0]]}]
task = Task("local", train=examples, test=examples)
assert task.to_dict() == {"id": "local", "train": examples, "test": examples}
assert task.scoreA([[1, 0]])
assert not task.scoreA([[0, 1]])
assert not task.scoreA([[1]])
assert fmt_grid(task.train[0][0], colour=False) == "0 1"
print("arckit: grid conversion, formatting and answer scoring passed")
