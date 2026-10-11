"""SerpApi client tests. Every network call is mocked; no credits are spent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from painter.config import SerpApiSettings
from painter.errors import (
    SerpApiAuthError,
    SerpApiBudgetExceeded,
    SerpApiNetworkError,
    SerpApiQuotaExceeded,
    SerpApiRateLimited,
    SerpApiResponseError,
)
from painter.serpapi_client import SerpApiClient, UsageLedger, describe_configuration


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise requests.RequestException("no more mocked responses")
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(tmp_path: Path, responses, **overrides) -> SerpApiClient:
    settings = SerpApiSettings(
        api_key="test-key",
        enabled=True,
        cache_enabled=overrides.pop("cache_enabled", True),
        max_retries=overrides.pop("max_retries", 0),
        backoff_base_s=0.001,
        **overrides,
    )
    client = SerpApiClient(settings, tmp_path / "cache", UsageLedger(tmp_path / "usage.json"), "doc_test")
    client._session = FakeSession(responses)
    return client


def test_missing_key_raises_auth_error(tmp_path):
    client = SerpApiClient(SerpApiSettings(api_key=None, enabled=True), tmp_path / "cache")
    client._session = FakeSession([])
    with pytest.raises(SerpApiAuthError):
        client.upload_image(b"jpeg")


def test_upload_and_search_happy_path(tmp_path):
    client = make_client(
        tmp_path,
        [
            FakeResponse(200, {"image_id": "abc123"}),
            FakeResponse(
                200,
                {
                    "text_results": [{"text": "INVOICE 2026", "link": "https://example.com"}],
                    "visual_matches": [{"title": "Logo", "image": "https://img/1.png", "image_width": 400, "image_height": 200}],
                    "exact_matches": [],
                },
            ),
        ],
    )
    payload = client.lens_search(b"jpeg-bytes")
    assert payload["text_results"][0]["text"] == "INVOICE 2026"
    texts = client.parse_text_results(payload)
    assert texts and texts[0]["origin"] == "text_results"
    matches = client.parse_image_matches(payload)
    assert matches[0]["url"] == "https://img/1.png"
    assert client.has_text_results(payload) is True
    assert client.usage.uploads == 1 and client.usage.searches == 1


def test_upload_rejects_oversized_payload(tmp_path):
    client = make_client(tmp_path, [], upload_max_bytes=10)
    with pytest.raises(SerpApiResponseError):
        client.upload_image(b"x" * 50)


def test_missing_image_id_is_reported(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, {"status": "ok"})])
    with pytest.raises(SerpApiResponseError):
        client.upload_image(b"jpeg")


def test_auth_error_classification(tmp_path):
    client = make_client(tmp_path, [FakeResponse(401, {"error": "Invalid API key"})])
    with pytest.raises(SerpApiAuthError):
        client.upload_image(b"jpeg")


def test_quota_error_classification(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, {"error": "Your account has run out of searches"})])
    with pytest.raises(SerpApiQuotaExceeded):
        client.upload_image(b"jpeg")


def test_rate_limit_is_retried_then_succeeds(tmp_path):
    client = make_client(
        tmp_path,
        [
            FakeResponse(429, {}, headers={"Retry-After": "1"}),
            FakeResponse(200, {"image_id": "xyz"}),
        ],
        max_retries=1,
    )
    assert client.upload_image(b"jpeg") == "xyz"
    assert client.usage.retries == 1


def test_network_failure_is_wrapped(tmp_path):
    client = make_client(tmp_path, [requests.Timeout("slow")], max_retries=0)
    with pytest.raises(SerpApiNetworkError):
        client.upload_image(b"jpeg")


def test_retries_exhausted_raises_rate_limit(tmp_path):
    client = make_client(tmp_path, [FakeResponse(429, {}), FakeResponse(429, {})], max_retries=1)
    with pytest.raises(SerpApiRateLimited):
        client.upload_image(b"jpeg")


def test_non_json_response(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, None, text="<html>oops</html>")])
    with pytest.raises(SerpApiResponseError):
        client.upload_image(b"jpeg")


def test_per_document_budget(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, {"image_id": "1"})], max_requests_per_document=1)
    assert client.upload_image(b"jpeg") == "1"
    with pytest.raises(SerpApiBudgetExceeded):
        client.upload_image(b"jpeg2")


def test_global_budget(tmp_path):
    ledger_path = tmp_path / "usage.json"
    ledger = UsageLedger(ledger_path)
    ledger.record("doc_a", "search", 5)
    settings = SerpApiSettings(api_key="k", enabled=True, max_requests_total=5, backoff_base_s=0.001)
    client = SerpApiClient(settings, tmp_path / "cache", ledger, "doc_b")
    client._session = FakeSession([FakeResponse(200, {"image_id": "1"})])
    with pytest.raises(SerpApiBudgetExceeded):
        client.upload_image(b"jpeg")


def test_cache_prevents_duplicate_requests(tmp_path):
    client = make_client(
        tmp_path,
        [FakeResponse(200, {"image_id": "abc"}), FakeResponse(200, {"text_results": [{"text": "hi"}]})],
    )
    first = client.lens_search(b"same-bytes")
    second = client.lens_search(b"same-bytes")
    assert first == second
    assert client.usage.cache_hits >= 1
    assert client.usage.uploads == 1  # the second call was served from cache


def test_cache_survives_new_client(tmp_path):
    client_a = make_client(tmp_path, [FakeResponse(200, {"image_id": "abc"}), FakeResponse(200, {"text_results": [{"text": "hi"}]})])
    client_a.lens_search(b"payload")
    client_b = make_client(tmp_path, [])
    assert client_b.lens_search(b"payload")["text_results"][0]["text"] == "hi"
    assert client_b.usage.uploads == 0


def test_parse_text_results_prefers_text_results_block(tmp_path):
    payload = {
        "text_results": [{"text": "REAL OCR"}],
        "organic_results": [{"title": "Search noise", "snippet": "ignore me"}],
    }
    entries = SerpApiClient.parse_text_results(payload)
    assert entries[0]["text"] == "REAL OCR"
    assert all(entry["origin"] != "organic_results" or entry["text"] != "REAL OCR" for entry in entries)


def test_parse_missing_fields_are_tolerated():
    assert SerpApiClient.parse_text_results({}) == []
    assert SerpApiClient.parse_image_matches({}) == []
    assert SerpApiClient.parse_text_results({"text_results": [{}]}) == []


def test_describe_configuration_hides_key():
    description = describe_configuration(SerpApiSettings(api_key="secret-value", enabled=True))
    assert "secret-value" not in json.dumps(description)
    assert description["key_configured"] is True
