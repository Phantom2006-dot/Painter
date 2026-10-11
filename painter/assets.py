"""Image/logo asset handling: extraction, matching and safe download.

Policy from the brief:

* the crop taken from the original document always wins when it is usable,
* online candidates are only a fallback and must be ranked, never trusted
  blindly,
* downloads are validated (scheme, host, size, content type) so the service can
  never be used to fetch arbitrary internal URLs,
* every asset records where it came from.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np

from .config import Settings
from .errors import AssetDownloadError, UnsafeUrlError
from .logging_utils import get_logger
from .preprocess import crop_box
from .schemas import AssetOrigin, BBox, Element, ImageCandidate, ImageContent

logger = get_logger("assets")

ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "metadata.google.internal"}


def perceptual_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Difference hash (dHash) as a hex string — robust to scaling/compression."""
    import cv2

    if image is None or image.size == 0:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if value else "0" for value in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    if not first or not second or len(first) != len(second):
        return 64
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def hash_similarity(first: str, second: str) -> float:
    return max(0.0, 1.0 - hamming_distance(first, second) / 64.0)


@dataclass
class AssetStore:
    """Writes extracted assets inside a document workspace."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_crop(self, image: np.ndarray, box: BBox, name: str, padding: int = 0) -> Path:
        import cv2

        crop = crop_box(image, box, padding=padding)
        path = self.root / f"{name}.png"
        if not cv2.imwrite(str(path), crop):
            raise AssetDownloadError(f"could not write asset {path}")
        return path

    def relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root.parent.resolve()))
        except ValueError:
            return str(path)


def extract_asset(
    image: np.ndarray,
    element: Element,
    store: AssetStore,
    *,
    padding: int = 0,
) -> ImageContent:
    """Always-available fallback: keep the crop from the original document."""
    path = store.save_crop(image, element.bbox, element.id, padding=padding)
    crop = crop_box(image, element.bbox, padding=padding)
    height, width = crop.shape[:2]
    return ImageContent(
        asset_path=store.relative(path),
        asset_origin=AssetOrigin.EXTRACTED,
        width_px=int(width),
        height_px=int(height),
        aspect_ratio=round(width / height, 4) if height else None,
        perceptual_hash=perceptual_hash(crop),
        notes=["cropped from the source document"],
    )


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"refusing to download non-http(s) URL: {parsed.scheme}")
    host = (parsed.hostname or "").lower()
    if not host or host in BLOCKED_HOSTNAMES:
        raise UnsafeUrlError(f"refusing to download from host {host!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"cannot resolve host {host!r}: {exc}") from exc
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise UnsafeUrlError(f"refusing to download from non-public address {address}")


def safe_download(url: str, destination: Path, settings: Settings) -> Path:
    """Download an image with SSRF protection, a size cap and validation."""
    import cv2
    import requests

    _validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(5, 20),
            headers={"User-Agent": "Painter/1.0 (document reconstruction)"},
        ) as response:
            if response.status_code != 200:
                raise AssetDownloadError(f"download failed with HTTP {response.status_code}")
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                raise AssetDownloadError(f"refusing non-image content type {content_type!r}")
            total = 0
            chunks: list[bytes] = []
            for chunk in response.iter_content(65536):
                total += len(chunk)
                if total > settings.asset_download_max_bytes:
                    raise AssetDownloadError("candidate image exceeds the configured size limit")
                chunks.append(chunk)
    except AssetDownloadError:
        raise
    except Exception as exc:  # noqa: BLE001 - network failures are expected
        raise AssetDownloadError(f"download failed: {exc}") from exc

    data = b"".join(chunks)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssetDownloadError("downloaded payload is not a decodable image")
    if not cv2.imwrite(str(destination), image):
        raise AssetDownloadError("could not store the downloaded image")
    return destination


def score_candidate(
    candidate: ImageCandidate, crop: np.ndarray, crop_hash: str, settings: Settings
) -> ImageCandidate:
    """Score a candidate against the original crop (0..1)."""
    import cv2

    crop_height, crop_width = crop.shape[:2]
    crop_aspect = crop_width / crop_height if crop_height else 0.0
    breakdown: dict[str, float] = {}

    candidate_aspect = 0.0
    if candidate.width and candidate.height:
        candidate_aspect = candidate.width / candidate.height
    if candidate_aspect and crop_aspect:
        aspect_score = 1.0 - min(1.0, abs(candidate_aspect - crop_aspect) / max(crop_aspect, 0.2))
    else:
        aspect_score = 0.5
    breakdown["aspect_ratio"] = round(max(0.0, aspect_score), 4)

    resolution_score = 0.5
    if candidate.width and candidate.height:
        short_side = min(candidate.width, candidate.height)
        if short_side >= max(crop_width, crop_height) * 0.8:
            resolution_score = 1.0
        elif short_side >= max(crop_width, crop_height) * 0.4:
            resolution_score = 0.7
        else:
            resolution_score = 0.3
    breakdown["resolution"] = round(resolution_score, 4)

    hash_score = 0.0
    local_path = candidate.local_path
    if local_path and Path(local_path).is_file():
        downloaded = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if downloaded is not None:
            hash_score = hash_similarity(crop_hash, perceptual_hash(downloaded))
    breakdown["perceptual_hash"] = round(hash_score, 4)

    match_kind_score = 1.0 if candidate.match_kind == "exact_matches" else 0.6
    breakdown["match_kind"] = match_kind_score

    weights = {"aspect_ratio": 0.25, "resolution": 0.2, "perceptual_hash": 0.4, "match_kind": 0.15}
    candidate.score = round(sum(breakdown[key] * weights[key] for key in weights), 4)
    candidate.score_breakdown = breakdown
    if candidate.score < settings.asset_match_threshold:
        candidate.rejected_reason = (
            f"score {candidate.score:.2f} below threshold {settings.asset_match_threshold:.2f}"
        )
    return candidate


def recover_image_asset(
    element: Element,
    page_image: np.ndarray,
    store: AssetStore,
    settings: Settings,
    serpapi_client: Any | None = None,
) -> ImageContent:
    """Extract the region, then (optionally) look for a better online asset."""
    content = extract_asset(page_image, element, store)
    if serpapi_client is None or not getattr(serpapi_client, "usable", False):
        content.notes.append("online image search disabled (SerpApi not enabled)")
        return content

    from .preprocess import encode_jpeg

    crop = crop_box(page_image, element.bbox)
    try:
        jpeg = encode_jpeg(crop, quality=90, max_bytes=settings.serpapi.upload_max_bytes)
        payload = serpapi_client.lens_search(jpeg)
    except Exception as exc:  # noqa: BLE001 - search failure must not break reconstruction
        content.notes.append(f"image search failed: {exc}")
        return content

    content.search_query_used = True
    raw_matches = serpapi_client.parse_image_matches(payload)
    candidates: list[ImageCandidate] = []
    for match in raw_matches:
        candidates.append(
            ImageCandidate(
                url=str(match.get("url") or ""),
                title=match.get("title"),
                source=match.get("source"),
                width=int(match["width"]) if str(match.get("width") or "").isdigit() else None,
                height=int(match["height"]) if str(match.get("height") or "").isdigit() else None,
                provider=match.get("provider") or "google_lens",
                license_note="external candidate; verify licensing before reuse",
            )
        )
    for candidate in candidates:
        candidate.score = 0.0
        candidate.score_breakdown = {}

    if not candidates:
        content.notes.append("no visual or exact matches returned by Google Lens")
        content.candidates = []
        return content

    # download only the most promising few, then score them against the crop
    for candidate in candidates[:4]:
        if settings.enable_assets_download and candidate.url:
            try:
                destination = store.root / f"{element.id}_candidate_{abs(hash(candidate.url)) % 10**8}.png"
                safe_download(candidate.url, destination, settings)
                candidate.local_path = store.relative(destination)
                candidate.downloaded = True
            except (AssetDownloadError, UnsafeUrlError) as exc:
                candidate.rejected_reason = str(exc)
        candidate = score_candidate(candidate, crop, content.perceptual_hash or "", settings)

    candidates.sort(key=lambda c: -c.score)
    content.candidates = candidates[:8]
    best = candidates[0]
    if best.downloaded and best.score >= settings.asset_match_threshold and best.local_path:
        # Only replace the extracted crop when the match is clearly better.
        content.asset_path = best.local_path
        content.asset_origin = AssetOrigin.RECOVERED
        content.source_url = best.url
        content.source_title = best.title
        content.matched = True
        content.match_confidence = best.score
        content.notes.append(
            f"replaced with an online match (score {best.score:.2f}); verify licensing before publishing"
        )
    else:
        content.matched = False
        content.match_confidence = best.score
        content.notes.append(
            f"kept the extracted crop; best online candidate scored {best.score:.2f}"
        )
    return content


def render_overlay(image: np.ndarray, elements: Sequence[Element], *, show_labels: bool = True) -> np.ndarray:
    """Draw detection boxes and labels on a copy of the page (UI preview)."""
    import cv2

    colors = {
        "heading": (255, 128, 0),
        "subheading": (255, 176, 32),
        "paragraph": (79, 195, 247),
        "text_block": (79, 195, 247),
        "list": (140, 220, 255),
        "table": (60, 200, 120),
        "image": (255, 90, 200),
        "logo": (255, 140, 220),
        "signature": (200, 120, 255),
        "stamp": (0, 120, 255),
        "header": (180, 180, 180),
        "footer": (180, 180, 180),
        "line": (120, 120, 120),
        "rectangle": (120, 120, 120),
        "checkbox": (0, 255, 255),
        "page_number": (200, 200, 200),
    }
    canvas = image.copy()
    for element in elements:
        color = colors.get(element.type.value, (200, 200, 200))
        box = element.bbox
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            color,
            2,
        )
        if show_labels:
            label = f"{element.type.value} {element.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(int(box.y1) - text_h - 6, 0)
            cv2.rectangle(
                canvas,
                (int(box.x1), top),
                (int(box.x1) + text_w + 6, top + text_h + 6),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (int(box.x1) + 3, top + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return canvas
