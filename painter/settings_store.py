"""Operator-changeable settings that survive a restart.

The SerpApi key is the one value an operator routinely pastes in by hand, so the
web UI can store it here instead of requiring an environment edit. The file lives
in the data directory with ``0600`` permissions, the key is never returned to a
client (only a masked hint) and it is never written to the log.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import SerpApiSettings, Settings
from .errors import ValidationError
from .logging_utils import get_logger

logger = get_logger("settings")

SETTINGS_FILENAME = "settings.json"
MAX_KEY_LENGTH = 256

#: Only these keys may be changed at runtime; everything else stays environment
#: driven so a stray API call cannot reconfigure the whole service.
EDITABLE_KEYS: tuple[str, ...] = ("serpapi_api_key", "serpapi_enabled")


def settings_path(data_dir: Path) -> Path:
    """Location of the runtime settings file."""
    return Path(data_dir) / SETTINGS_FILENAME


def load_overrides(data_dir: Path) -> dict[str, Any]:
    """Return the stored overrides, ignoring an unreadable file."""
    path = settings_path(data_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text() or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable settings file %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if key in EDITABLE_KEYS}


def save_overrides(data_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the stored overrides and write them with 0600.

    ``None`` removes a value. Unknown keys are ignored rather than stored.
    """
    path = settings_path(data_dir)
    current = load_overrides(data_dir)
    for key, value in updates.items():
        if key not in EDITABLE_KEYS:
            continue
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)  # the file holds a credential
    temporary.replace(path)
    return current


def mask_secret(value: str | None) -> str | None:
    """A hint that lets an operator recognise a key without revealing it."""
    if not value:
        return None
    tail = value[-4:] if len(value) > 4 else ""
    return f"••••{tail}"


def validate_api_key(value: str) -> str:
    """Reject values that cannot be a SerpApi key before storing them."""
    key = (value or "").strip()
    if not key:
        raise ValidationError("the SerpApi key is empty")
    if len(key) > MAX_KEY_LENGTH:
        raise ValidationError("the SerpApi key is implausibly long")
    if any(character.isspace() for character in key):
        raise ValidationError("the SerpApi key must not contain whitespace")
    return key


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Return a copy of ``settings`` with the stored overrides applied."""
    if not overrides:
        return settings
    changes: dict[str, Any] = {}
    if "serpapi_api_key" in overrides:
        changes["api_key"] = overrides["serpapi_api_key"] or None
    if "serpapi_enabled" in overrides:
        changes["enabled"] = bool(overrides["serpapi_enabled"])
    if not changes:
        return settings
    return replace(settings, serpapi=replace(settings.serpapi, **changes))


def describe(settings: Settings, data_dir: Path) -> dict[str, Any]:
    """Client-safe state: never includes the key itself."""
    stored = load_overrides(data_dir)
    key = settings.serpapi.api_key
    source = None
    if key:
        source = "settings file" if stored.get("serpapi_api_key") else "environment"
    return {
        "serpapi_key_set": bool(key),
        "serpapi_key_hint": mask_secret(key),
        "serpapi_key_source": source,
        "serpapi_enabled": bool(settings.serpapi.enabled),
        "serpapi_base_url": settings.serpapi.base_url,
        "ocr_provider": settings.ocr_provider,
    }


def verify_api_key(api_key: str, base_url: str, timeout_s: float = 20.0) -> tuple[bool, str]:
    """Check a key against SerpApi's account endpoint.

    Returns ``(ok, message)``; the message never echoes the key.
    """
    import requests

    url = f"{base_url.rstrip('/')}/account"
    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - any transport failure is a "no"
        return False, f"could not reach SerpApi: {exc}"
    try:
        payload = response.json()
    except ValueError:
        return False, f"SerpApi returned a non-JSON response (HTTP {response.status_code})"
    if isinstance(payload, dict) and payload.get("error"):
        return False, str(payload["error"])
    if response.status_code == 200 and isinstance(payload, dict):
        account = payload.get("account_email") or payload.get("plan_name")
        return True, f"key accepted for {account}" if account else "key accepted"
    return False, f"SerpApi rejected the key (HTTP {response.status_code})"


__all__ = [
    "EDITABLE_KEYS",
    "SETTINGS_FILENAME",
    "SerpApiSettings",
    "apply_overrides",
    "describe",
    "load_overrides",
    "mask_secret",
    "save_overrides",
    "settings_path",
    "validate_api_key",
    "verify_api_key",
]
