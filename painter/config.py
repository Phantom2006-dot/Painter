"""Central configuration for Painter.

Every tunable is read from the environment (optionally from a ``.env`` file at
the repository root) so that no credential or path is ever hardcoded. Secrets
are never logged: see :mod:`painter.logging_utils`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL_DIR = REPO_ROOT / "yolo_demo"


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (no third-party dependency).

    Existing environment variables always win, so exported values are never
    silently overwritten by a stale file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(REPO_ROOT / ".env")


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class SerpApiSettings:
    """SerpApi / Google Lens configuration.

    ``enabled`` is deliberately opt-in: the API is metered, so the pipeline only
    calls it when an operator turns it on (``PAINTER_ENABLE_SERPAPI=true``).
    """

    api_key: str | None = None
    base_url: str = "https://serpapi.com"
    timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
    max_retries: int = 3
    backoff_base_s: float = 1.5
    backoff_max_s: float = 20.0
    max_requests_per_document: int = 30
    max_requests_total: int = 400
    cache_enabled: bool = True
    upload_max_bytes: int = 480_000
    search_max_bytes: int = 900
    enabled: bool = False
    #: Google Lens ``type`` parameter used for image recovery searches.
    lens_type: str = "all"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def usable(self) -> bool:
        return self.enabled and self.configured


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the whole application."""

    data_dir: Path = DEFAULT_DATA_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    max_upload_mb: int = 25
    allowed_image_formats: tuple[str, ...] = ("png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp")
    allowed_document_formats: tuple[str, ...] = ("pdf",)
    target_dpi: int = 200
    max_page_pixels: int = 12_000_000
    max_pages_per_document: int = 20

    enable_yolo: bool = True
    yolo_layout_model: Path | None = None
    yolo_coco_model: Path | None = None
    yolo_custom_model: Path | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 1024
    yolo_device: str | None = None

    ocr_provider: str = "auto"  # auto | tesseract | lens | none
    tesseract_lang: str = "eng"
    ocr_min_crop_px: int = 12
    ocr_max_crop_px: int = 2200
    ocr_upscale_small_crops: bool = True

    enable_llm: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0

    enable_assets_download: bool = False
    asset_download_max_bytes: int = 8_000_000
    asset_match_threshold: float = 0.72

    verify_enabled: bool = True
    verify_target_similarity: float = 0.93
    verify_max_iterations: int = 3
    verify_min_improvement: float = 0.004
    rasterize_dpi: int = 150

    api_token: str | None = None
    job_workers: int = 2
    log_level: str = "INFO"
    serpapi: SerpApiSettings = field(default_factory=SerpApiSettings)

    # -- derived helpers -------------------------------------------------
    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "painter.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.documents_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_model(name: str | None, default_filename: str, model_dir: Path) -> Path | None:
    if name:
        candidate = Path(name).expanduser()
        if not candidate.is_absolute():
            candidate = (model_dir / candidate).resolve()
        return candidate if candidate.is_file() else None
    candidate = model_dir / default_filename
    return candidate if candidate.is_file() else None


def load_settings(**overrides: object) -> Settings:
    """Build :class:`Settings` from the environment, applying ``overrides``.

    ``refresh=True`` re-reads ``.env`` and the environment before building, so a
    test (or a long-lived process) can pick up a changed configuration. The flag
    is consumed here and never passed on to :class:`Settings`.
    """
    if overrides.pop("refresh", False):
        _load_dotenv(REPO_ROOT / ".env")
    model_dir = Path(_env_str("PAINTER_MODEL_DIR", str(DEFAULT_MODEL_DIR)) or DEFAULT_MODEL_DIR)
    serpapi = SerpApiSettings(
        api_key=_env_str("SERPAPI_KEY") or _env_str("PAINTER_SERPAPI_KEY"),
        base_url=_env_str("PAINTER_SERPAPI_BASE_URL", "https://serpapi.com") or "https://serpapi.com",
        timeout_s=_env_float("PAINTER_SERPAPI_TIMEOUT_S", 30.0),
        connect_timeout_s=_env_float("PAINTER_SERPAPI_CONNECT_TIMEOUT_S", 10.0),
        max_retries=_env_int("PAINTER_SERPAPI_MAX_RETRIES", 3),
        backoff_base_s=_env_float("PAINTER_SERPAPI_BACKOFF_BASE_S", 1.5),
        backoff_max_s=_env_float("PAINTER_SERPAPI_BACKOFF_MAX_S", 20.0),
        max_requests_per_document=_env_int("PAINTER_SERPAPI_MAX_REQUESTS_PER_DOC", 30),
        max_requests_total=_env_int("PAINTER_SERPAPI_MAX_REQUESTS_TOTAL", 400),
        cache_enabled=_env_bool("PAINTER_SERPAPI_CACHE", True),
        upload_max_bytes=_env_int("PAINTER_SERPAPI_UPLOAD_MAX_BYTES", 480_000),
        search_max_bytes=_env_int("PAINTER_SERPAPI_SEARCH_MAX_BYTES", 900),
        enabled=_env_bool("PAINTER_ENABLE_SERPAPI", False),
        lens_type=_env_str("PAINTER_SERPAPI_LENS_TYPE", "all") or "all",
    )

    settings = Settings(
        data_dir=Path(_env_str("PAINTER_DATA_DIR", str(DEFAULT_DATA_DIR)) or DEFAULT_DATA_DIR),
        model_dir=model_dir,
        max_upload_mb=_env_int("PAINTER_MAX_UPLOAD_MB", 25),
        target_dpi=_env_int("PAINTER_TARGET_DPI", 200),
        max_page_pixels=_env_int("PAINTER_MAX_PAGE_PIXELS", 12_000_000),
        max_pages_per_document=_env_int("PAINTER_MAX_PAGES", 20),
        enable_yolo=_env_bool("PAINTER_ENABLE_YOLO", True),
        yolo_layout_model=_resolve_model(_env_str("PAINTER_LAYOUT_MODEL"), "doclayout.pt", model_dir),
        yolo_coco_model=_resolve_model(_env_str("PAINTER_COCO_MODEL"), "yolov8n.pt", model_dir),
        yolo_custom_model=_resolve_model(_env_str("PAINTER_CUSTOM_MODEL"), "painter_doclayout.pt", model_dir),
        yolo_conf=_env_float("PAINTER_YOLO_CONF", 0.25),
        yolo_iou=_env_float("PAINTER_YOLO_IOU", 0.45),
        yolo_imgsz=_env_int("PAINTER_YOLO_IMGSZ", 1024),
        yolo_device=_env_str("PAINTER_YOLO_DEVICE"),
        ocr_provider=_env_str("PAINTER_OCR_PROVIDER", "auto") or "auto",
        tesseract_lang=_env_str("PAINTER_TESSERACT_LANG", "eng") or "eng",
        ocr_upscale_small_crops=_env_bool("PAINTER_OCR_UPSCALE", True),
        enable_llm=_env_bool("PAINTER_ENABLE_LLM", False),
        llm_model=_env_str("PAINTER_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        llm_timeout_s=_env_float("PAINTER_LLM_TIMEOUT_S", 60.0),
        enable_assets_download=_env_bool("PAINTER_ENABLE_ASSET_DOWNLOAD", False),
        asset_download_max_bytes=_env_int("PAINTER_ASSET_MAX_BYTES", 8_000_000),
        asset_match_threshold=_env_float("PAINTER_ASSET_MATCH_THRESHOLD", 0.72),
        verify_enabled=_env_bool("PAINTER_VERIFY", True),
        verify_target_similarity=_env_float("PAINTER_VERIFY_TARGET", 0.93),
        verify_max_iterations=_env_int("PAINTER_VERIFY_MAX_ITERATIONS", 3),
        verify_min_improvement=_env_float("PAINTER_VERIFY_MIN_IMPROVEMENT", 0.004),
        rasterize_dpi=_env_int("PAINTER_RASTERIZE_DPI", 150),
        api_token=_env_str("PAINTER_API_TOKEN"),
        job_workers=_env_int("PAINTER_JOB_WORKERS", 2),
        log_level=_env_str("PAINTER_LOG_LEVEL", "INFO") or "INFO",
        serpapi=serpapi,
    )
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})
    return settings


_CACHED: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return a process-wide settings singleton."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = load_settings()
    return _CACHED
