"""Shared explicit artifact selection for port guest acceptance."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def guest_factory():
    """Create a guest using artifact paths supplied by the acceptance harness."""

    def create(*, bundle_env, requirements=(), universe_env=None, wheel_envs=(), **limits):
        keys = [bundle_env, *wheel_envs]
        if requirements:
            keys.extend([universe_env, "SHELLSIM_PATCHED_UV"])
        if any(key is None or not os.environ.get(key) for key in keys):
            pytest.skip("set port CPython bundle, package catalog and patched uv paths")
        from ports._support.testing import mount_guest

        return mount_guest(
            Path(os.environ[bundle_env]),
            universe=Path(os.environ[universe_env]) if universe_env else None,
            uv=Path(os.environ["SHELLSIM_PATCHED_UV"]) if requirements else None,
            requirements=requirements,
            wheels=[Path(os.environ[key]) for key in wheel_envs],
            **limits,
        )

    return create
