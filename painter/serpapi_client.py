"""SerpApi / Google Lens client.

Design goals taken from the project brief:

* the API key only ever comes from the environment and is never logged,
* request timeouts, retries with exponential backoff and jitter,
* response validation with explicit handling of missing fields,
* a configurable usage budget (per document and globally) plus on-disk caching
  so the same crop is never searched twice,
* clear, typed errors for authentication, quota, rate limiting and network
  failures.

Google Lens is *not* treated as a universal OCR engine here: ``text_results``
are parsed when the plan exposes them, but the provider built on top of this
client labels such text as search-derived rather than line-level OCR.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import SerpApiSettings
from .errors import (
    SerpApiAuthError,
    SerpApiBudgetExceeded,
    SerpApiError,
    SerpApiNetworkError,
    SerpApiQuotaExceeded,
    SerpApiRateLimited,
    SerpApiResponseError,
)
from .logging_utils import get_logger, redact

logger = get_logger("serpapi")

QUOTA_MARKERS = (
    "run out of searches",
    "out of searches",
    "no searches left",
    "quota",
    "plan limit",
)
AUTH_MARKERS = ("invalid api key", "invalid key", "unauthorized", "missing api key", "api key")


@dataclass
class SerpApiUsage:
    """Counters describing how much API budget has been spent."""

    uploads: int = 0
    searches: int = 0
    cache_hits: int = 0
    retries: int = 0
    errors: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uploads": self.uploads,
            "searches": self.searches,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "errors": self.errors,
            "by_kind": dict(self.by_kind),
            "total_requests": self.uploads + self.searches,
        }


class UsageLedger:
    """Persistent, process-safe counter of SerpApi calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"total_requests": 0, "documents": {}}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def record(self, document_id: str, kind: str, count: int = 1) -> int:
        with self._lock:
            data = self._read()
            data["total_requests"] = int(data.get("total_requests", 0)) + count
            documents = data.setdefault("documents", {})
            entry = documents.setdefault(document_id, {"total_requests": 0, "by_kind": {}})
            entry["total_requests"] = int(entry.get("total_requests", 0)) + count
            entry["by_kind"][kind] = int(entry["by_kind"].get(kind, 0)) + count
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp.replace(self.path)
            return data["total_requests"]


class SerpApiClient:
    """Thin, defensive wrapper over the two SerpApi endpoints we need."""

    def __init__(
        self,
        settings: SerpApiSettings,
        cache_dir: Path,
        ledger: UsageLedger | None = None,
        document_id: str = "unknown",
    ) -> None:
        self.settings = settings
        self.cache_dir = Path(cache_dir) / "serpapi"
        self.ledger = ledger or UsageLedger(Path(cache_dir) / "serpapi_usage.json")
        self.document_id = document_id
        self.usage = SerpApiUsage()
        self._session = requests.Session()
        self._memo: dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.settings.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public helpers --------------------------------------------------
    @property
    def usable(self) -> bool:
        return self.settings.usable

    def usage_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self.usage.as_dict())
        snapshot["global"] = self.ledger.snapshot().get("total_requests", 0)
        snapshot["budget_per_document"] = self.settings.max_requests_per_document
        snapshot["budget_total"] = self.settings.max_requests_total
        return snapshot

    # -- budget ----------------------------------------------------------
    def _check_budget(self) -> None:
        if self.usage.uploads + self.usage.searches >= self.settings.max_requests_per_document:
            raise SerpApiBudgetExceeded(
                f"per-document SerpApi budget of {self.settings.max_requests_per_document} "
                "requests is exhausted",
                details={"usage": self.usage.as_dict()},
            )
        total = self.ledger.snapshot().get("total_requests", 0)
        if total >= self.settings.max_requests_total:
            raise SerpApiBudgetExceeded(
                f"global SerpApi budget of {self.settings.max_requests_total} requests is exhausted",
                details={"global_requests": total},
            )

    def _charge(self, kind: str) -> None:
        if kind == "upload":
            self.usage.uploads += 1
        else:
            self.usage.searches += 1
        self.usage.by_kind[kind] = self.usage.by_kind.get(kind, 0) + 1
        self.ledger.record(self.document_id, kind)

    # -- caching ---------------------------------------------------------
    def _cache_key(self, kind: str, payload: bytes, params: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(kind.encode())
        digest.update(payload)
        digest.update(json.dumps(params, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key in self._memo:
                self.usage.cache_hits += 1
                return self._memo[key]
        if not self.settings.cache_enabled:
            return None
        path = self._cache_path(key)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self._memo[key] = data
                    self.usage.cache_hits += 1
                return data
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._memo[key] = value
        if not self.settings.cache_enabled:
            return
        try:
            path = self._cache_path(key)
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - cache is best effort
            logger.debug("cache write failed: %s", exc)

    # -- HTTP ------------------------------------------------------------
    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        kind: str = "search",
    ) -> dict[str, Any]:
        if not self.settings.configured:
            raise SerpApiAuthError(
                "SerpApi is not configured: set the SERPAPI_KEY environment variable"
            )
        self._check_budget()

        params = dict(params or {})
        data = dict(data or {})
        params["api_key"] = self.settings.api_key
        if data:
            data["api_key"] = self.settings.api_key

        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.settings.max_retries:
            try:
                response = self._session.request(
                    method,
                    url,
                    params=params,
                    files=files,
                    data=data,
                    timeout=(self.settings.connect_timeout_s, self.settings.timeout_s),
                )
            except requests.Timeout as exc:
                last_error = SerpApiNetworkError(f"SerpApi request timed out: {exc}")
            except requests.RequestException as exc:
                last_error = SerpApiNetworkError(f"SerpApi request failed: {exc}")
            else:
                status = response.status_code
                if status in (401, 403):
                    raise SerpApiAuthError(
                        "SerpApi rejected the API key (HTTP %d)" % status,
                        details={"status": status},
                    )
                if status == 429:
                    retry_after = response.headers.get("Retry-After")
                    last_error = SerpApiRateLimited(
                        "SerpApi rate limit reached",
                        details={"retry_after": retry_after},
                    )
                elif status >= 500:
                    last_error = SerpApiNetworkError(f"SerpApi server error (HTTP {status})")
                elif status >= 400:
                    raise SerpApiResponseError(
                        f"SerpApi returned HTTP {status}",
                        details={"body": redact(response.text[:300])},
                    )
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise SerpApiResponseError(
                            f"SerpApi returned non-JSON content: {redact(response.text[:200])}"
                        ) from exc
                    error_message = str(payload.get("error") or "")
                    if error_message:
                        lowered = error_message.lower()
                        if any(marker in lowered for marker in AUTH_MARKERS) and "key" in lowered:
                            raise SerpApiAuthError(error_message)
                        if any(marker in lowered for marker in QUOTA_MARKERS):
                            raise SerpApiQuotaExceeded(error_message)
                        raise SerpApiError(error_message, details={"payload_keys": list(payload)})
                    self._charge(kind)
                    return payload

            attempt += 1
            if attempt > self.settings.max_retries:
                break
            self.usage.retries += 1
            delay = min(
                self.settings.backoff_base_s * (2 ** (attempt - 1)),
                self.settings.backoff_max_s,
            ) + random.uniform(0, 0.4)
            logger.warning(
                "SerpApi %s failed (attempt %d/%d): %s - retrying in %.1fs",
                kind,
                attempt,
                self.settings.max_retries,
                last_error,
                delay,
            )
            time.sleep(delay)

        self.usage.errors += 1
        raise last_error or SerpApiError("SerpApi request failed for an unknown reason")

    # -- endpoints -------------------------------------------------------
    def upload_image(self, jpeg_bytes: bytes, cache_key_extra: str = "") -> str:
        """Upload an image and return the ``image_id`` used by Lens."""
        if len(jpeg_bytes) > self.settings.upload_max_bytes:
            raise SerpApiResponseError(
                f"image is {len(jpeg_bytes)} bytes, above the "
                f"{self.settings.upload_max_bytes} byte upload cap"
            )
        key = self._cache_key("upload", jpeg_bytes, {"extra": cache_key_extra})
        cached = self._cache_get(key)
        if cached and cached.get("image_id"):
            return str(cached["image_id"])
        payload = self._request(
            "POST",
            f"{self.settings.base_url}/image",
            files={"image": ("crop.jpg", jpeg_bytes, "image/jpeg")},
            kind="upload",
        )
        image_id = payload.get("image_id")
        if not image_id:
            raise SerpApiResponseError(
                "SerpApi image upload did not return an image_id",
                details={"keys": list(payload)},
            )
        self._cache_put(key, {"image_id": image_id})
        return str(image_id)

    def lens_search(
        self,
        image_bytes: bytes,
        *,
        hl: str = "en",
        lens_type: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a Google Lens search for an image and return the raw payload."""
        params: dict[str, Any] = {
            "engine": "google_lens",
            "hl": hl,
            "type": lens_type or self.settings.lens_type,
        }
        if extra_params:
            params.update(extra_params)
        key = self._cache_key("lens", image_bytes, params)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        image_id = self.upload_image(image_bytes)
        request_params = {k: v for k, v in params.items()}
        request_params["image_id"] = image_id
        payload = self._request(
            "GET",
            f"{self.settings.base_url}/search.json",
            params=request_params,
            kind="search",
        )
        self._cache_put(key, payload)
        return payload

    # -- parsing ---------------------------------------------------------
    @staticmethod
    def parse_text_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect text-bearing results, preserving provenance.

        ``text_results`` (added by SerpApi in August 2026) is preferred; organic
        results and ``ai_overview`` are only used as fallbacks because they are
        search results, not OCR of the submitted crop.
        """
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(text: str | None, origin: str, link: str | None = None, title: str | None = None) -> None:
            if not text:
                return
            cleaned = " ".join(str(text).split())
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            collected.append({"text": cleaned, "origin": origin, "link": link, "title": title})

        for item in payload.get("text_results") or []:
            if isinstance(item, dict):
                add(item.get("text") or item.get("title"), "text_results", item.get("link"), item.get("title"))
            elif isinstance(item, str):
                add(item, "text_results")

        for item in payload.get("organic_results") or []:
            if isinstance(item, dict):
                add(item.get("snippet") or item.get("title"), "organic_results", item.get("link"), item.get("title"))

        ai_overview = payload.get("ai_overview")
        if isinstance(ai_overview, dict):
            add(ai_overview.get("answer") or ai_overview.get("content"), "ai_overview")
        return collected

    @staticmethod
    def parse_image_matches(payload: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
        """Extract visual and exact matches with metadata for ranking."""
        matches: list[dict[str, Any]] = []
        for kind in ("exact_matches", "visual_matches"):
            for item in payload.get(kind) or []:
                if not isinstance(item, dict):
                    continue
                link = item.get("link") or item.get("image") or item.get("thumbnail")
                if not link:
                    continue
                image_info = item.get("image") if isinstance(item.get("image"), dict) else {}
                matches.append(
                    {
                        "url": item.get("image") or item.get("link"),
                        "page_url": item.get("link"),
                        "thumbnail": item.get("thumbnail"),
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "width": item.get("image_width") or (image_info or {}).get("width"),
                        "height": item.get("image_height") or (image_info or {}).get("height"),
                        "match_kind": kind,
                        "provider": "google_lens",
                    }
                )
                if len(matches) >= limit:
                    return matches
        return matches

    @staticmethod
    def has_text_results(payload: dict[str, Any]) -> bool:
        return bool(payload.get("text_results"))


# --------------------------------------------------------------------------
# small helper used by tests and the CLI
# --------------------------------------------------------------------------
def describe_configuration(settings: SerpApiSettings) -> dict[str, Any]:
    """Describe SerpApi configuration without leaking the key."""
    return {
        "enabled": settings.enabled,
        "key_configured": settings.configured,
        "base_url": settings.base_url,
        "cache_enabled": settings.cache_enabled,
        "timeout_s": settings.timeout_s,
        "max_retries": settings.max_retries,
        "budget_per_document": settings.max_requests_per_document,
        "budget_total": settings.max_requests_total,
    }
