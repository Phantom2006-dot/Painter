"""Structured logging helpers.

Log records are JSON objects on stderr/stdout, which makes them easy to grep and
safe to ship to a log collector. A redaction filter removes anything that looks
like an API key so credentials never reach the logs.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_SECRET_PATTERNS = [
    re.compile(r"([?&](?:api_key|apikey|key|token)=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"\b([0-9a-f]{32,})\b", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*\"?)([^\"\s,}]+)"),
]

REDACTED = "[redacted]"


def redact(value: Any) -> Any:
    """Return ``value`` with anything resembling a credential removed."""
    if isinstance(value, str):
        out = value
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, out)
        return out
    if isinstance(value, dict):
        return {k: ("[redacted]" if _is_secret_key(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("api_key", "apikey", "token", "secret", "password"))


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact(v) for k, v in record.args.items()}
            else:
                record.args = tuple(redact(a) for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("stage", "document_id", "job_id", "element_id", "duration_ms", "provider"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(redact(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger("painter")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_RedactingFilter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"painter.{name}" if not name.startswith("painter.") else name)
    if not logger.handlers and not logging.getLogger("painter").handlers:
        configure_logging()
    return logger


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``msg`` with structured fields attached."""
    logger.log(level, msg, extra={"extra_fields": fields})


@contextmanager
def timed(logger: logging.Logger, stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log its duration, even when it raises."""
    start = time.perf_counter()
    outcome: dict[str, Any] = {}
    try:
        yield outcome
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            f"stage {stage} finished",
            stage=stage,
            duration_ms=duration_ms,
            **{**fields, **outcome},
        )
