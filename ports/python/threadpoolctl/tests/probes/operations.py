"""Inspect the available guest providers, including an empty selection."""

from threadpoolctl import threadpool_info, threadpool_limits

assert isinstance(threadpool_info(), list)
with threadpool_limits(limits=1, user_api="blas"):
    assert isinstance(threadpool_info(), list)
print("threadpoolctl: guest provider discovery and limit context passed")
