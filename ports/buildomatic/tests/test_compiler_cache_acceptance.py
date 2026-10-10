"""The optional integration test uses a real installed/pinned cache and host cc."""

import os
from pathlib import Path

import pytest

from ports.buildomatic.acceptance.compiler_cache import CacheCounters, _client_environment, local_probe


def test_probe_clients_accept_only_public_daemon_endpoint(tmp_path):
    endpoint = {"SCCACHE_SERVER_PORT": "4226"}
    environment = _client_environment(endpoint, tmp_path)
    assert environment["SCCACHE_CLIENT_SIDE"] == "1"
    assert "SCCACHE_BASEDIRS" not in environment
    assert "SCCACHE_CLIENT_SIDE" not in _client_environment(endpoint, tmp_path, client_side=False)
    for secret in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "SCCACHE_BUCKET"):
        with pytest.raises(ValueError):
            _client_environment({**endpoint, secret: "private"}, tmp_path)
        assert secret not in environment


def test_counter_delta_rejects_daemon_reset_and_counts_exact_events():
    baseline = CacheCounters(2, 3, 3, 3, 0)
    assert CacheCounters(3, 3, 3, 3, 0) - baseline == CacheCounters(1, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        CacheCounters(0, 0, 0, 0, 0) - baseline


def test_real_host_c_compile_hit_link_and_flag_invalidation(tmp_path, monkeypatch):
    executable = os.environ.get("BUILDOMATIC_SCCACHE")
    if not executable:
        pytest.skip("set BUILDOMATIC_SCCACHE to the pinned executable for real compiler-cache acceptance")
    # Ambient credentials must not become part of the daemon or compiler probe.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "acceptance-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "acceptance-sentinel")
    result = local_probe(Path(executable), Path("/usr/bin/cc"), tmp_path / "probe")
    assert result.first == CacheCounters(0, 1, 1, 1, 0)
    assert result.second == CacheCounters(1, 0, 0, 0, 0)
    assert result.changed_flags == CacheCounters(0, 1, 1, 1, 0)
    assert result.link == CacheCounters(0, 0, 0, 0, 0)
    assert result.local_missing in (CacheCounters(0, 1, 1, 1, 0), CacheCounters(0, 1, 1, 1, 1))
    assert result.local_corrupt in (CacheCounters(0, 1, 1, 1, 0), CacheCounters(0, 1, 1, 1, 1))
    assert result.object_sha256 != result.changed_object_sha256


@pytest.mark.parametrize("debug", [False, True])
def test_real_upstream_wasm_compile_hit_and_stable_debug(tmp_path, debug):
    executable = os.environ.get("BUILDOMATIC_SCCACHE")
    compiler = os.environ.get("BUILDOMATIC_PROBE_WASM_CLANG")
    if not executable or not compiler:
        pytest.skip("set pinned sccache and existing upstream Wasm Clang for isolated acceptance")
    result = local_probe(
        Path(executable), Path(compiler), tmp_path / "probe", target="wasm", debug=debug, containment=True
    )
    assert result.first == CacheCounters(0, 1, 1, 1, 0)
    assert result.second == CacheCounters(1, 0, 0, 0, 0)
    assert result.changed_flags == CacheCounters(0, 1, 1, 1, 0)
    assert result.link == CacheCounters(0, 0, 0, 0, 0)
    assert result.local_missing in (CacheCounters(0, 1, 1, 1, 0), CacheCounters(0, 1, 1, 1, 1))
    assert result.local_corrupt in (CacheCounters(0, 1, 1, 1, 0), CacheCounters(0, 1, 1, 1, 1))
    assert result.object_sha256 != result.changed_object_sha256
    assert result.client_side
    assert result.containment.compiler_in_action_group
    assert result.containment.cancelled
    assert result.containment.independent_compile_succeeded
