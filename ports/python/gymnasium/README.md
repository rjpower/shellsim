# Gymnasium

Adapts the official Gymnasium wheel to load AsyncVectorEnv only when requested,
so unavailable process semaphores do not block synchronous environments. The
guest probe checks seeded headless CartPole and FrozenLake behavior, termination,
and SyncVectorEnv batching and automatic resets. AsyncVectorEnv requires
unavailable process semaphores; rendering and optional extras are not qualified.
Upstream metadata and extras are preserved.
