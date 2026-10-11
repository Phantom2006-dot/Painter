"""Runtime settings: the SerpApi key can be pasted in and must stay secret."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from painter.api import create_app
from painter.config import load_settings
from painter.errors import ValidationError
from painter.settings_store import (
    describe,
    load_overrides,
    mask_secret,
    save_overrides,
    validate_api_key,
)

KEY = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_settings_state_never_returns_the_key(client, settings):
    response = client.put("/api/settings/serpapi", json={"api_key": KEY, "enabled": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["serpapi_key_set"] is True
    assert payload["serpapi_enabled"] is True
    assert payload["serpapi_key_hint"] == mask_secret(KEY)
    # the key itself must appear nowhere in the response
    assert KEY not in json.dumps(payload)

    state = client.get("/api/settings").json()
    assert state["serpapi_key_hint"] == mask_secret(KEY)
    assert KEY not in json.dumps(state)


def test_put_applies_the_key_to_the_running_pipeline(client, settings):
    client.put("/api/settings/serpapi", json={"api_key": KEY, "enabled": True})
    live = client.app.state.pipeline.settings
    assert live.serpapi.api_key == KEY
    assert live.serpapi.enabled is True
    # and it survives a fresh load, because it is stored on disk
    assert load_settings(refresh=True).serpapi.api_key == KEY


def test_settings_file_is_private(settings):
    save_overrides(settings.data_dir, {"serpapi_api_key": KEY, "serpapi_enabled": True})
    path = settings.data_dir / "settings.json"
    assert path.is_file()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"settings file is {oct(mode)}"


def test_delete_forgets_the_key(client, settings):
    client.put("/api/settings/serpapi", json={"api_key": KEY, "enabled": True})
    response = client.delete("/api/settings/serpapi")
    assert response.status_code == 200
    payload = response.json()
    assert payload["serpapi_key_set"] is False
    assert payload["serpapi_enabled"] is False
    assert load_settings(refresh=True).serpapi.api_key is None
    stored = load_overrides(settings.data_dir)
    assert "serpapi_api_key" not in stored  # the secret is gone from disk
    assert stored.get("serpapi_enabled") is False


def test_verify_uses_the_account_endpoint(client, monkeypatch):
    calls = {}

    def fake_verify(api_key, base_url, timeout_s=20.0):
        calls["api_key"] = api_key
        return False, "Invalid API key"

    monkeypatch.setattr("painter.api.verify_api_key", fake_verify)
    response = client.put(
        "/api/settings/serpapi", json={"api_key": KEY, "enabled": True, "verify": True}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is False
    assert payload["message"] == "Invalid API key"
    assert calls["api_key"] == KEY
    # a rejected key is still stored: the operator can retry without retyping
    assert payload["serpapi_key_set"] is True


def test_verify_without_a_key_is_a_validation_error(client):
    response = client.post("/api/settings/serpapi/verify")
    assert response.status_code == 400
    assert "no SerpApi key" in response.json()["message"]


def test_bad_keys_are_rejected(client):
    assert client.put("/api/settings/serpapi", json={}).status_code == 400
    assert client.put("/api/settings/serpapi", json={"api_key": "   "}).status_code == 400
    spaced = client.put("/api/settings/serpapi", json={"api_key": "abc def"})
    assert spaced.status_code == 400
    assert "whitespace" in spaced.json()["message"]


def test_validate_and_mask_helpers():
    with pytest.raises(ValidationError):
        validate_api_key("")
    with pytest.raises(ValidationError):
        validate_api_key("x" * 300)
    assert mask_secret(None) is None
    assert mask_secret("abcd") == "••••"
    assert mask_secret("0123456789") == "••••6789"


def test_describe_reports_where_the_key_came_from(settings):
    from dataclasses import replace

    environment_key = replace(settings, serpapi=replace(settings.serpapi, api_key=KEY))
    described = describe(environment_key, settings.data_dir)
    assert described["serpapi_key_source"] == "environment"
    assert described["serpapi_key_hint"] == mask_secret(KEY)
    save_overrides(settings.data_dir, {"serpapi_api_key": KEY})
    stored = describe(environment_key, settings.data_dir)
    assert stored["serpapi_key_source"] == "settings file"


def test_unreadable_settings_file_is_ignored(settings, caplog):
    path = Path(settings.data_dir) / "settings.json"
    path.write_text("{not json")
    assert load_overrides(settings.data_dir) == {}
    # a corrupt file must not break settings loading
    assert load_settings(refresh=True) is not None


def test_unknown_keys_are_not_stored(settings):
    save_overrides(settings.data_dir, {"serpapi_api_key": KEY, "data_dir": "/etc"})
    stored = load_overrides(settings.data_dir)
    assert stored == {"serpapi_api_key": KEY}
