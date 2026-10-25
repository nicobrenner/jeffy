"""Tests for the optional PyPI update check.

No test here touches the network or the real ~/.jeffy/ directory.

Run:
    JEFFY_BACKEND=torch python -m pytest tests/test_update_check.py -v
"""

import json
import time

import pytest

from jeffy import update_check
from jeffy.update_check import _is_newer, _version_key, check_for_update


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point the cache at tmp_path and clear opt-out vars for determinism."""
    monkeypatch.delenv("JEFFY_NO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(update_check, "CACHE_PATH", str(tmp_path / ".jeffy" / "update_check.json"))
    return tmp_path


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test hits the HTTP path when it should not."""
    calls = []

    def _forbidden(timeout):
        calls.append(timeout)
        raise AssertionError("_fetch_latest should not have been called")

    monkeypatch.setattr(update_check, "_fetch_latest", _forbidden)
    return calls


def _stub_fetch(monkeypatch, value):
    calls = []

    def _fetch(timeout):
        calls.append(timeout)
        return value

    monkeypatch.setattr(update_check, "_fetch_latest", _fetch)
    return calls


class TestVersionComparison:
    def test_prerelease_ordering(self):
        assert _version_key("0.1.0a16") < _version_key("0.1.0a17")
        assert _version_key("0.1.0a17") < _version_key("0.1.0")

    def test_is_newer_prereleases(self):
        assert _is_newer("0.1.0a17", "0.1.0a16")
        assert _is_newer("0.1.0", "0.1.0a17")
        assert not _is_newer("0.1.0a16", "0.1.0a17")
        assert not _is_newer("0.1.0a17", "0.1.0a17")

    def test_fallback_matches_packaging(self, monkeypatch):
        # Force the tuple fallback by making the packaging import fail.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _no_packaging(name, *args, **kwargs):
            if name.startswith("packaging"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_packaging)
        assert _is_newer("0.1.0a17", "0.1.0a16")
        assert not _is_newer("0.1.0a16", "0.1.0a17")
        assert _is_newer("0.2.0", "0.1.0")

    def test_multi_digit_and_release_segments(self):
        assert _version_key("0.1.0a9") < _version_key("0.1.0a16")
        assert _version_key("0.1.0") < _version_key("0.2.0")
        assert _version_key("1.0.0") > _version_key("0.9.9")


class TestCheckForUpdate:
    def test_newer_version_available(self, monkeypatch):
        _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"

    def test_same_version_returns_none(self, monkeypatch):
        _stub_fetch(monkeypatch, "0.1.0a17")
        assert check_for_update("0.1.0a17") is None

    def test_older_version_on_pypi_returns_none(self, monkeypatch):
        _stub_fetch(monkeypatch, "0.1.0a15")
        assert check_for_update("0.1.0a17") is None

    def test_network_error_returns_none(self, monkeypatch):
        def _boom(timeout):
            raise OSError("dns failure")

        monkeypatch.setattr(update_check, "_fetch_latest", _boom)
        assert check_for_update("0.1.0a17") is None

    def test_fetch_returning_none_is_safe(self, monkeypatch):
        _stub_fetch(monkeypatch, None)
        assert check_for_update("0.1.0a17") is None

    def test_unparseable_remote_version_returns_none(self, monkeypatch):
        _stub_fetch(monkeypatch, "not-a-version")
        assert check_for_update("0.1.0a17") is None


class TestFetchLatestParsing:
    """_fetch_latest swallows malformed payloads; urlopen is always mocked."""

    def _mock_urlopen(self, monkeypatch, payload: bytes):
        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return payload

        monkeypatch.setattr(update_check.urllib.request, "urlopen", lambda *a, **k: _Resp())

    def test_valid_payload(self, monkeypatch):
        self._mock_urlopen(monkeypatch, json.dumps({"info": {"version": "0.1.0a18"}}).encode())
        assert update_check._fetch_latest(2.0) == "0.1.0a18"

    def test_malformed_json(self, monkeypatch):
        self._mock_urlopen(monkeypatch, b"{not json")
        assert update_check._fetch_latest(2.0) is None

    def test_missing_info_version(self, monkeypatch):
        self._mock_urlopen(monkeypatch, json.dumps({"info": {}}).encode())
        assert update_check._fetch_latest(2.0) is None

    def test_timeout_raises_nothing(self, monkeypatch):
        def _timeout(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(update_check.urllib.request, "urlopen", _timeout)
        assert update_check._fetch_latest(2.0) is None


class TestOptOut:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_jeffy_no_update_check_blocks_everything(self, monkeypatch, no_network, value):
        monkeypatch.setenv("JEFFY_NO_UPDATE_CHECK", value)
        assert check_for_update("0.1.0a17") is None
        assert no_network == []

    def test_falsy_opt_out_still_checks(self, monkeypatch):
        monkeypatch.setenv("JEFFY_NO_UPDATE_CHECK", "0")
        calls = _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"
        assert len(calls) == 1

    def test_ci_env_blocks_check(self, monkeypatch, no_network):
        monkeypatch.setenv("CI", "true")
        assert check_for_update("0.1.0a17") is None
        assert no_network == []

    def test_opt_out_writes_no_cache(self, monkeypatch, isolated_env):
        monkeypatch.setenv("JEFFY_NO_UPDATE_CHECK", "1")
        check_for_update("0.1.0a17")
        assert not (isolated_env / ".jeffy" / "update_check.json").exists()


class TestCache:
    def test_fresh_cache_skips_network(self, monkeypatch, isolated_env, no_network):
        cache = isolated_env / ".jeffy" / "update_check.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"last_check": int(time.time()), "latest": "0.1.0a18"}))
        assert check_for_update("0.1.0a17") == "0.1.0a18"
        assert no_network == []

    def test_fresh_cache_with_same_version_returns_none(self, monkeypatch, isolated_env, no_network):
        cache = isolated_env / ".jeffy" / "update_check.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"last_check": int(time.time()), "latest": "0.1.0a17"}))
        assert check_for_update("0.1.0a17") is None
        assert no_network == []

    def test_stale_cache_hits_network(self, monkeypatch, isolated_env):
        cache = isolated_env / ".jeffy" / "update_check.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        stale = int(time.time()) - (update_check.CACHE_TTL_SECONDS + 60)
        cache.write_text(json.dumps({"last_check": stale, "latest": "0.1.0a17"}))
        calls = _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"
        assert len(calls) == 1

    def test_corrupt_cache_falls_back_to_network(self, monkeypatch, isolated_env):
        cache = isolated_env / ".jeffy" / "update_check.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("{{{ not json at all")
        calls = _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"
        assert len(calls) == 1

    def test_cache_missing_keys_falls_back(self, monkeypatch, isolated_env):
        cache = isolated_env / ".jeffy" / "update_check.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"something": "else"}))
        calls = _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"
        assert len(calls) == 1

    def test_successful_check_writes_cache(self, monkeypatch, isolated_env):
        _stub_fetch(monkeypatch, "0.1.0a18")
        check_for_update("0.1.0a17")
        data = json.loads((isolated_env / ".jeffy" / "update_check.json").read_text())
        assert data["latest"] == "0.1.0a18"
        assert isinstance(data["last_check"], int)

    def test_unwritable_cache_dir_is_survivable(self, monkeypatch):
        def _boom(*a, **k):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(update_check.Path, "mkdir", _boom)
        _stub_fetch(monkeypatch, "0.1.0a18")
        assert check_for_update("0.1.0a17") == "0.1.0a18"


class TestLogUpdateNotice:
    def test_logs_when_update_available(self, monkeypatch, caplog):
        monkeypatch.setattr(update_check, "installed_version", lambda: "0.1.0a17")
        monkeypatch.setattr(update_check, "check_for_update", lambda current, timeout=2.0: "0.1.0a18")
        with caplog.at_level("INFO", logger="jeffy.update_check"):
            update_check.log_update_notice()
        assert "Update available: 0.1.0a17 → 0.1.0a18" in caplog.text
        assert "pip install -U jeffy-classify" in caplog.text

    def test_silent_when_up_to_date(self, monkeypatch, caplog):
        monkeypatch.setattr(update_check, "installed_version", lambda: "0.1.0a17")
        monkeypatch.setattr(update_check, "check_for_update", lambda current, timeout=2.0: None)
        with caplog.at_level("INFO", logger="jeffy.update_check"):
            update_check.log_update_notice()
        assert caplog.text == ""

    def test_silent_when_version_unknown(self, monkeypatch, caplog):
        monkeypatch.setattr(update_check, "installed_version", lambda: None)
        with caplog.at_level("INFO", logger="jeffy.update_check"):
            update_check.log_update_notice()
        assert caplog.text == ""

    def test_never_raises(self, monkeypatch):
        def _boom():
            raise RuntimeError("unexpected")

        monkeypatch.setattr(update_check, "installed_version", _boom)
        update_check.log_update_notice()


class TestServerStartupResilience:
    def test_startup_survives_failing_update_check(self, monkeypatch):
        from fastapi.testclient import TestClient

        from jeffy import server

        def _boom():
            raise RuntimeError("update check exploded")

        monkeypatch.setattr("jeffy.update_check.log_update_notice", _boom)
        with TestClient(server.app) as client:
            assert client.get("/health").status_code == 200

    def test_start_update_check_swallows_thread_errors(self, monkeypatch):
        from jeffy import server

        def _boom(*a, **k):
            raise RuntimeError("cannot start thread")

        monkeypatch.setattr("threading.Thread", _boom)
        server._start_update_check()
