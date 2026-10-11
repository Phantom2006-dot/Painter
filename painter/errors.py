"""Error taxonomy shared by every Painter subsystem.

Each error carries a machine readable ``code`` plus a ``retryable`` flag so the
pipeline can decide whether a failed stage may be retried without operator
intervention.
"""

from __future__ import annotations

from typing import Any


class PainterError(Exception):
    """Base class for all Painter errors.

    ``code`` is a stable machine-readable identifier, ``retryable`` tells the
    pipeline whether a failed stage may be retried unattended, and
    ``status_code`` is the HTTP status the API returns for it.
    """

    code = "painter_error"
    retryable = False
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


# -- input / validation ---------------------------------------------------
class UploadValidationError(PainterError):
    code = "upload_invalid"
    status_code = 400


class UnsupportedFormatError(UploadValidationError):
    code = "unsupported_format"
    status_code = 415


class UploadTooLargeError(UploadValidationError):
    code = "upload_too_large"
    status_code = 413


class PageLimitExceededError(UploadValidationError):
    code = "page_limit_exceeded"
    status_code = 413


class NotFoundError(PainterError):
    code = "not_found"
    status_code = 404


class UnauthorizedError(PainterError):
    code = "unauthorized"
    status_code = 401


class ValidationError(PainterError):
    """A request or edit payload failed validation."""

    code = "validation_failed"
    status_code = 400


class ConflictError(PainterError):
    code = "conflict"
    status_code = 409


# -- processing -----------------------------------------------------------
class PreprocessingError(PainterError):
    code = "preprocessing_failed"


class CorruptImageError(PreprocessingError):
    code = "corrupt_image"
    status_code = 422


class DetectionError(PainterError):
    code = "detection_failed"


class ModelUnavailableError(DetectionError):
    code = "model_unavailable"
    status_code = 503


class OcrError(PainterError):
    code = "ocr_failed"
    retryable = True


class OcrUnavailableError(OcrError):
    code = "ocr_unavailable"
    status_code = 503


class OcrEmptyResult(OcrError):
    code = "ocr_empty"


class StructureError(PainterError):
    code = "structure_failed"


class GeometryError(PainterError):
    code = "geometry_failed"


class TypographyError(PainterError):
    code = "typography_failed"


class TableReconstructionError(PainterError):
    code = "table_reconstruction_failed"


class RenderingError(PainterError):
    code = "rendering_failed"


class RasterizationError(RenderingError):
    code = "rasterization_failed"


class VerificationError(PainterError):
    code = "verification_failed"


class StorageError(PainterError):
    code = "storage_failed"


# -- SerpApi --------------------------------------------------------------
class SerpApiError(PainterError):
    code = "serpapi_error"


class SerpApiAuthError(SerpApiError):
    code = "serpapi_auth_error"
    status_code = 502


class SerpApiQuotaExceeded(SerpApiError):
    code = "serpapi_quota_exceeded"
    status_code = 429


class SerpApiRateLimited(SerpApiError):
    code = "serpapi_rate_limited"
    retryable = True
    status_code = 429


class SerpApiNetworkError(SerpApiError):
    code = "serpapi_network_error"
    retryable = True
    status_code = 504


class SerpApiResponseError(SerpApiError):
    code = "serpapi_response_error"


class SerpApiBudgetExceeded(SerpApiError):
    code = "serpapi_budget_exceeded"
    status_code = 429


class AssetDownloadError(PainterError):
    code = "asset_download_failed"
    retryable = True


class UnsafeUrlError(AssetDownloadError):
    code = "unsafe_url"
    status_code = 400


# -- pipeline -------------------------------------------------------------
class PipelineError(PainterError):
    code = "pipeline_failed"


class StageFailed(PipelineError):
    code = "stage_failed"


class JobCancelled(PipelineError):
    code = "job_cancelled"
    status_code = 409
