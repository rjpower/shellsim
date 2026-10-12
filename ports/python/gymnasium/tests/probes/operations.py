"""Exercise upstream headless environments, seeds and synchronous vector reset."""

import gymnasium as gym
import numpy as np

first = gym.make("CartPole-v1")
second = gym.make("CartPole-v1")
initial, _ = first.reset(seed=42)
repeated, _ = second.reset(seed=42)
assert initial.dtype == np.float32 and initial.shape == (4,)
np.testing.assert_array_equal(initial, repeated)
for action in (0, 1, 0, 1, 1, 0):
    observation, reward, terminated, truncated, _ = first.step(action)
    replay, replay_reward, replay_terminated, replay_truncated, _ = second.step(action)
    np.testing.assert_array_equal(observation, replay)
    assert (reward, terminated, truncated) == (replay_reward, replay_terminated, replay_truncated)
    assert reward == 1.0 and not terminated and not truncated
assert not np.array_equal(observation, initial)
first.reset(seed=42)
for _ in range(100):
    _, reward, terminated, truncated, _ = first.step(0)
    if terminated:
        break
else:
    raise AssertionError("CartPole failed to terminate after sustained left force")
assert reward == 1.0 and not truncated
first.close()
second.close()

lake = gym.make("FrozenLake-v1", is_slippery=False)
assert lake.reset(seed=42)[0] == 0
for action, expected in zip((2, 2, 1, 1, 1, 2), (1, 2, 6, 10, 14, 15)):
    observation, reward, terminated, truncated, info = lake.step(action)
    assert observation == expected and info["prob"] == 1.0 and not truncated
    assert reward == (1.0 if expected == 15 else 0.0)
    assert terminated == (expected == 15)
lake.close()

first = gym.make("FrozenLake-v1", is_slippery=True)
second = gym.make("FrozenLake-v1", is_slippery=True)
assert first.reset(seed=19)[0] == second.reset(seed=19)[0] == 0
for action in (2, 1, 2, 1, 1, 2, 0, 3):
    result = first.step(action)
    assert result == second.step(action)
    if result[2] or result[3]:
        break
first.close()
second.close()

vector = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1", max_episode_steps=1) for _ in range(2)])
individuals = [gym.make("CartPole-v1", max_episode_steps=1) for _ in range(2)]
observations, _ = vector.reset(seed=42)
assert observations.shape == (2, 4) and observations.dtype == np.float32
for index, environment in enumerate(individuals):
    np.testing.assert_array_equal(observations[index], environment.reset(seed=42 + index)[0])
observations, rewards, terminated, truncated, _ = vector.step(np.array([0, 1]))
for index, environment in enumerate(individuals):
    expected = environment.step(index)
    np.testing.assert_array_equal(observations[index], expected[0])
    assert (rewards[index], terminated[index], truncated[index]) == expected[1:4]
assert rewards.tolist() == [1.0, 1.0] and truncated.tolist() == [True, True]
_, rewards, terminated, truncated, _ = vector.step(np.array([0, 1]))
assert rewards.tolist() == [0.0, 0.0] and not terminated.any() and not truncated.any()
vector.close()
for environment in individuals:
    environment.close()

print("gymnasium: seeded headless and SyncVectorEnv operations passed")
