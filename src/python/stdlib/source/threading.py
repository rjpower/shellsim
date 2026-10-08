"""Single-interpreter lock surface for code that guards in-process state.

Python thread creation is not modeled. These locks only coordinate code running in the
one Python interpreter; they grant no host thread or process capability.
"""


class _RLock:
    def __init__(self):
        self._depth = 0

    def acquire(self, blocking=True, timeout=-1):
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout < -1:
            raise ValueError("timeout value must be positive")
        self._depth += 1
        return True

    def release(self):
        if self._depth == 0:
            raise RuntimeError("cannot release un-acquired lock")
        self._depth -= 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

    def _is_owned(self):
        return self._depth > 0


def RLock():
    """Return a reentrant lock for the current simulated interpreter."""
    return _RLock()
