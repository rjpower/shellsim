"""Opt-in host timings; never use or write the guest clock or output streams."""

import logging
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger("shellsim.startup")


@contextmanager
def phase(name: str, **details: object):
    """Log nested setup/execution phases when the host enables INFO logging.

    Parent durations include their children. Start records make slow or failed
    operations visible without logging guest source, URLs, or credentials.
    """
    if not logger.isEnabledFor(logging.INFO):
        yield
        return
    description = " ".join([name, *(f"{key}={value}" for key, value in details.items())])
    logger.info("%s: start", description)
    start = perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        logger.info("%s: %s in %.3fs", description, "complete" if completed else "failed", perf_counter() - start)
