"""Generate and grade fixed small tasks through the public dataset factory."""

import reasoning_gym

for name in ("leg_counting", "basic_arithmetic"):
    dataset = reasoning_gym.create_dataset(name, size=3, seed=42)
    entry = dataset[0]
    assert entry == dataset[0]
    assert len(dataset) == 3
    assert entry["question"] and entry["answer"] and entry["metadata"]
    assert dataset.score_answer(entry["answer"], entry) == 1.0
    assert dataset.score_answer("invalid answer", entry) == 0.0
try:
    reasoning_gym.create_dataset("missing_dataset", size=1, seed=42)
except ValueError:
    pass
else:
    raise AssertionError("unknown dataset accepted")
print("reasoning-gym: deterministic generation, grading and unknown task passed")
