"""Optional check for a newer release on PyPI.

Never blocks, never raises, and never runs when opted out. Jeffy is a local
tool, so the check is cached for a day and can be disabled entirely with
JEFFY_NO_UPDATE_CHECK=1.
"""

import json
import logging
import os
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/jeffy-classify/json"
PACKAGE_NAME = "jeffy-classify"
CACHE_PATH = "~/.jeffy/update_check.json"
CACHE_TTL_SECONDS = 24 * 60 * 60

_FALSY = {"", "0", "false", "no"}
_TRUTHY = {"1", "true", "yes"}
_PRERELEASE_ORDER = {"a": 0, "b": 1, "rc": 2}


def _is_opted_out() -> bool:
    if os.environ.get("JEFFY_NO_UPDATE_CHECK", "").strip().lower() in _TRUTHY:
        return True
    return os.environ.get("CI", "").strip().lower() not in _FALSY


def _version_key(version: str) -> tuple:
    """PEP 440 ordering for the subset of versions Jeffy publishes.

    A prerelease sorts before its own final release, so the release segment
    carries a 0/1 flag: 0.1.0a16 < 0.1.0a17 < 0.1.0.
    """
    text = version.strip()
    pre_kind, pre_num = "", 0
    for kind in ("rc", "a", "b"):
        marker = text.find(kind)
        if marker > 0 and text[marker - 1].isdigit():
            suffix = text[marker + len(kind):]
            if suffix.isdigit():
                pre_kind, pre_num = kind, int(suffix)
                text = text[:marker]
                break
    release = tuple(int(part) for part in text.split("."))
    if pre_kind:
        return (release, 0, _PRERELEASE_ORDER[pre_kind], pre_num)
    return (release, 1, 0, 0)


def _is_newer(candidate: str, current: str) -> bool:
    try:
        from packaging.version import parse

        return parse(candidate) > parse(current)
    except Exception:
        pass
    return _version_key(candidate) > _version_key(current)


def _cache_file() -> Path:
    return Path(CACHE_PATH).expanduser()


def _read_cache() -> dict | None:
    try:
        with open(_cache_file()) as f:
            data = json.load(f)
        if isinstance(data, dict) and "last_check" in data and "latest" in data:
            return data
    except Exception:
        pass
    return None


def _write_cache(latest: str) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"last_check": int(time.time()), "latest": latest}, f)
    except Exception:
        pass


def _fetch_latest(timeout: float) -> str | None:
    """Return info.version from PyPI, or None on any failure."""
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = data["info"]["version"]
        return latest if isinstance(latest, str) and latest.strip() else None
    except Exception:
        return None


def check_for_update(current_version: str, timeout: float = 2.0) -> str | None:
    """Return the newer version available on PyPI, or None."""
    try:
        if _is_opted_out():
            return None

        cached = _read_cache()
        if cached is not None:
            try:
                age = time.time() - float(cached["last_check"])
            except Exception:
                age = CACHE_TTL_SECONDS + 1
            if 0 <= age < CACHE_TTL_SECONDS:
                latest = cached["latest"]
                return latest if _is_newer(latest, current_version) else None

        latest = _fetch_latest(timeout)
        if not latest:
            return None
        _write_cache(latest)
        return latest if _is_newer(latest, current_version) else None
    except Exception:
        return None


def installed_version() -> str | None:
    try:
        from importlib.metadata import version

        return version(PACKAGE_NAME)
    except Exception:
        return None


def log_update_notice(timeout: float = 2.0) -> None:
    """Log one line if a newer release exists. Silent otherwise."""
    try:
        current = installed_version()
        if not current:
            return
        latest = check_for_update(current, timeout=timeout)
        if latest:
            logger.info(
                f"Update available: {current} → {latest} "
                f"(pip install -U {PACKAGE_NAME})"
            )
    except Exception:
        pass
