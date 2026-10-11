"""Persistence: document workspaces on disk plus a SQLite repository.

Large artefacts (page images, crops, rendered files, diff maps) live on the file
system under ``data/documents/<document_id>/``; the database stores metadata,
stage state, metrics and provenance. The schema is documented in
``docs/ARCHITECTURE.md`` and can be moved to PostgreSQL by swapping the
repository implementation, since the pipeline only talks to this interface.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import NotFoundError, StorageError
from .logging_utils import get_logger

logger = get_logger("storage")

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    stored_path   TEXT NOT NULL,
    sha256        TEXT,
    content_type  TEXT,
    size_bytes    INTEGER DEFAULT 0,
    page_count    INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'created',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pages (
    document_id   TEXT NOT NULL,
    page_index    INTEGER NOT NULL,
    width_px      INTEGER DEFAULT 0,
    height_px     INTEGER DEFAULT 0,
    dpi           REAL DEFAULT 0,
    page_format   TEXT,
    orientation   TEXT,
    image_path    TEXT,
    original_path TEXT,
    preview_path  TEXT,
    metadata_json TEXT DEFAULT '{}',
    PRIMARY KEY (document_id, page_index)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    status       TEXT NOT NULL,
    stage        TEXT,
    progress     REAL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    options_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS job_stages (
    job_id       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    duration_ms  REAL,
    error        TEXT,
    details_json TEXT DEFAULT '{}',
    PRIMARY KEY (job_id, stage)
);

CREATE TABLE IF NOT EXISTS ir_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    revision     INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    json         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    class_name   TEXT,
    element_type TEXT,
    confidence   REAL,
    bbox         TEXT,
    source       TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    element_key  TEXT,
    provider     TEXT,
    method       TEXT,
    text         TEXT,
    confidence   REAL,
    flags        TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    element_id   TEXT,
    asset_path   TEXT,
    origin       TEXT,
    source_url   TEXT,
    matched      INTEGER DEFAULT 0,
    confidence   REAL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL,
    element_id      TEXT,
    url             TEXT,
    score           REAL,
    downloaded      INTEGER DEFAULT 0,
    local_path      TEXT,
    rejected_reason TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS typography (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    element_id   TEXT,
    json         TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iterations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    iteration    INTEGER,
    params_json  TEXT,
    metrics_json TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    metrics_json TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    format       TEXT,
    path         TEXT,
    size_bytes   INTEGER,
    created_at   TEXT NOT NULL,
    meta_json    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edits (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    element_id   TEXT,
    field        TEXT,
    old_value    TEXT,
    new_value    TEXT,
    editor       TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT,
    job_id       TEXT,
    stage        TEXT,
    code         TEXT,
    message      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_detections_document ON detections(document_id, page_index);
CREATE INDEX IF NOT EXISTS idx_ocr_document ON ocr_results(document_id, page_index);
CREATE INDEX IF NOT EXISTS idx_ir_document ON ir_versions(document_id, revision);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentWorkspace:
    """File-system layout for one document."""

    root: Path
    document_id: str

    SUBDIRECTORIES = ("source", "pages", "crops", "assets", "rendered", "verify", "stages", "exports")

    def __post_init__(self) -> None:
        # Construction is a pure lookup: creating directories here silently
        # recreated a workspace that ``cleanup_workspaces`` had just deleted.
        self.root = Path(self.root)

    def ensure(self) -> "DocumentWorkspace":
        """Create the directory layout for a new document."""
        for name in self.SUBDIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def dir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, kind: str, name: str) -> Path:
        return self.dir(kind) / name

    def relative(self, path: Path | str) -> str:
        """Return a workspace-relative POSIX path (used in the IR and the API)."""
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(candidate)

    def absolute(self, relative_path: str) -> Path:
        """Resolve a workspace-relative path, refusing traversal outside it."""
        candidate = (self.root / relative_path).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise StorageError(f"path escapes the document workspace: {relative_path}")
        return candidate


class Repository:
    """SQLite-backed metadata store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # -- documents -------------------------------------------------------
    def create_document(
        self,
        document_id: str,
        filename: str,
        stored_path: str,
        sha256: str,
        content_type: str,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utcnow()
        self.execute(
            """INSERT OR REPLACE INTO documents
               (document_id, filename, stored_path, sha256, content_type, size_bytes,
                page_count, status, created_at, updated_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, 0, 'created', ?, ?, ?)""",
            (
                document_id,
                filename,
                stored_path,
                sha256,
                content_type,
                size_bytes,
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )

    def update_document(self, document_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"page_count", "status", "metadata_json", "filename", "content_type", "size_bytes"}
        assignments = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(json.dumps(value) if key == "metadata_json" and not isinstance(value, str) else value)
        if not assignments:
            return
        assignments.append("updated_at = ?")
        values.append(utcnow())
        values.append(document_id)
        self.execute(f"UPDATE documents SET {', '.join(assignments)} WHERE document_id = ?", values)

    def get_document(self, document_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        if not rows:
            raise NotFoundError(f"document {document_id} not found")
        document = rows[0]
        document["metadata"] = json.loads(document.get("metadata_json") or "{}")
        return document

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def delete_document(self, document_id: str) -> None:
        for table in (
            "pages",
            "detections",
            "ocr_results",
            "assets",
            "image_candidates",
            "typography",
            "iterations",
            "quality_metrics",
            "ir_versions",
            "exports",
            "edits",
            "errors",
        ):
            self.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))
        self.execute("DELETE FROM jobs WHERE document_id = ?", (document_id,))
        self.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    # -- pages -----------------------------------------------------------
    def upsert_page(self, document_id: str, page: dict[str, Any]) -> None:
        self.execute(
            """INSERT OR REPLACE INTO pages
               (document_id, page_index, width_px, height_px, dpi, page_format, orientation,
                image_path, original_path, preview_path, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                page.get("page_index", 0),
                page.get("width_px", 0),
                page.get("height_px", 0),
                page.get("dpi", 0),
                page.get("page_format"),
                page.get("orientation"),
                page.get("image_path"),
                page.get("original_path"),
                page.get("preview_path"),
                json.dumps(page.get("metadata", {})),
            ),
        )

    def get_pages(self, document_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_index", (document_id,)
        )

    # -- jobs ------------------------------------------------------------
    def create_job(self, job_id: str, document_id: str, options: dict[str, Any] | None = None) -> None:
        self.execute(
            """INSERT OR REPLACE INTO jobs (job_id, document_id, status, stage, progress,
               created_at, options_json) VALUES (?, ?, 'queued', NULL, 0, ?, ?)""",
            (job_id, document_id, utcnow(), json.dumps(options or {})),
        )

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "stage", "progress", "error", "started_at", "finished_at", "options_json"}
        assignments = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(value)
        if not assignments:
            return
        values.append(job_id)
        self.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            raise NotFoundError(f"job {job_id} not found")
        job = rows[0]
        job["options"] = json.loads(job.get("options_json") or "{}")
        job["stages"] = self.get_job_stages(job_id)
        return job

    def get_jobs_for_document(self, document_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM jobs WHERE document_id = ? ORDER BY created_at DESC LIMIT ?",
            (document_id, limit),
        )

    def record_stage(
        self,
        job_id: str,
        stage: str,
        status: str,
        duration_ms: float | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.execute(
            """INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, duration_ms, error, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id, stage) DO UPDATE SET
                 status=excluded.status, finished_at=excluded.finished_at,
                 duration_ms=excluded.duration_ms, error=excluded.error, details_json=excluded.details_json""",
            (
                job_id,
                stage,
                status,
                utcnow(),
                utcnow(),
                duration_ms,
                error,
                json.dumps(details or {}, default=str),
            ),
        )

    def get_job_stages(self, job_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM job_stages WHERE job_id = ? ORDER BY started_at", (job_id,))

    def completed_stages(self, job_id: str) -> set[str]:
        rows = self.query(
            "SELECT stage FROM job_stages WHERE job_id = ? AND status = 'completed'", (job_id,)
        )
        return {row["stage"] for row in rows}

    # -- IR --------------------------------------------------------------
    def save_ir(self, document_id: str, revision: int, payload: str) -> None:
        self.execute(
            "INSERT INTO ir_versions (document_id, revision, created_at, json) VALUES (?, ?, ?, ?)",
            (document_id, revision, utcnow(), payload),
        )

    def latest_ir(self, document_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT * FROM ir_versions WHERE document_id = ? ORDER BY revision DESC LIMIT 1",
            (document_id,),
        )
        if not rows:
            return None
        return json.loads(rows[0]["json"])

    # -- analysis records -------------------------------------------------
    def save_detections(self, document_id: str, page_index: int, detections: Iterable[dict[str, Any]]) -> None:
        now = utcnow()
        rows = [
            (
                document_id,
                page_index,
                detection.get("class_name"),
                detection.get("type"),
                detection.get("confidence"),
                json.dumps(detection.get("bbox")),
                detection.get("source"),
                now,
            )
            for detection in detections
        ]
        if not rows:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                """INSERT INTO detections (document_id, page_index, class_name, element_type,
                   confidence, bbox, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def save_ocr(self, document_id: str, page_index: int, key: str, result: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO ocr_results (document_id, page_index, element_key, provider, method,
               text, confidence, flags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                page_index,
                key,
                result.get("provider"),
                result.get("method"),
                result.get("text"),
                result.get("confidence"),
                json.dumps(result.get("flags", [])),
                utcnow(),
            ),
        )

    def save_asset(self, document_id: str, element_id: str, content: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO assets (document_id, element_id, asset_path, origin, source_url,
               matched, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                element_id,
                content.get("asset_path"),
                content.get("asset_origin"),
                content.get("source_url"),
                1 if content.get("matched") else 0,
                content.get("match_confidence"),
                utcnow(),
            ),
        )
        for candidate in content.get("candidates", []) or []:
            self.execute(
                """INSERT INTO image_candidates (document_id, element_id, url, score, downloaded,
                   local_path, rejected_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    element_id,
                    candidate.get("url"),
                    candidate.get("score"),
                    1 if candidate.get("downloaded") else 0,
                    candidate.get("local_path"),
                    candidate.get("rejected_reason"),
                    utcnow(),
                ),
            )

    def save_typography(self, document_id: str, element_id: str, payload: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO typography (document_id, element_id, json, created_at) VALUES (?, ?, ?, ?)",
            (document_id, element_id, json.dumps(payload), utcnow()),
        )

    def save_iterations(self, document_id: str, page_index: int, iterations: list[dict[str, Any]]) -> None:
        for iteration in iterations:
            self.execute(
                """INSERT INTO iterations (document_id, page_index, iteration, params_json,
                   metrics_json, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    page_index,
                    iteration.get("iteration"),
                    json.dumps({k: v for k, v in iteration.items() if k.endswith("_mm") or k == "font_scale"}),
                    json.dumps(iteration),
                    utcnow(),
                ),
            )

    def save_quality(self, document_id: str, page_index: int, metrics: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO quality_metrics (document_id, page_index, metrics_json, created_at) VALUES (?, ?, ?, ?)",
            (document_id, page_index, json.dumps(metrics), utcnow()),
        )

    def record_export(self, document_id: str, format_name: str, path: str, size_bytes: int, meta: dict[str, Any] | None = None) -> None:
        self.execute(
            """INSERT INTO exports (document_id, format, path, size_bytes, created_at, meta_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (document_id, format_name, path, size_bytes, utcnow(), json.dumps(meta or {})),
        )

    def list_exports(self, document_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM exports WHERE document_id = ? ORDER BY created_at DESC", (document_id,)
        )

    def record_edit(
        self, document_id: str, element_id: str, field_name: str, old_value: Any, new_value: Any, editor: str = "user"
    ) -> None:
        self.execute(
            """INSERT INTO edits (document_id, element_id, field, old_value, new_value, editor, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                element_id,
                field_name,
                json.dumps(old_value, default=str),
                json.dumps(new_value, default=str),
                editor,
                utcnow(),
            ),
        )

    def record_error(
        self,
        document_id: str | None,
        job_id: str | None,
        stage: str | None,
        code: str,
        message: str,
    ) -> None:
        self.execute(
            """INSERT INTO errors (document_id, job_id, stage, code, message, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (document_id, job_id, stage, code, message[:2000], utcnow()),
        )

    def list_errors(self, document_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM errors WHERE document_id = ? ORDER BY created_at DESC LIMIT ?",
            (document_id, limit),
        )

    def stats(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for table in ("documents", "jobs", "pages", "detections", "exports", "errors"):
            rows = self.query(f"SELECT COUNT(*) AS count FROM {table}")
            result[table] = rows[0]["count"] if rows else 0
        return result


def remove_document_files(documents_dir: Path, document_id: str) -> bool:
    """Delete a document's workspace from disk, returning whether it existed.

    ``Repository.delete_document`` only clears database rows, so deleting a
    document used to leave its pages, crops, renders and exports on disk — 640MB
    of them on a single test machine. Deleting a document has to mean deleting
    its files.
    """
    import shutil

    workspace = Path(documents_dir) / document_id
    if not workspace.is_dir():
        return False
    shutil.rmtree(workspace, ignore_errors=True)
    return True


def cleanup_workspaces(
    documents_dir: Path, repository: Repository, keep_days: int = 30, dry_run: bool = False
) -> list[str]:
    """Delete workspaces whose documents are older than ``keep_days``."""
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    removed: list[str] = []
    if not documents_dir.is_dir():
        return removed
    for workspace in sorted(documents_dir.iterdir()):
        if not workspace.is_dir():
            continue
        try:
            rows = repository.query(
                "SELECT created_at FROM documents WHERE document_id = ?", (workspace.name,)
            )
            created = (
                datetime.fromisoformat(rows[0]["created_at"]) if rows else datetime.fromtimestamp(
                    workspace.stat().st_mtime, tz=timezone.utc
                )
            )
        except Exception:
            continue
        if created < cutoff:
            removed.append(workspace.name)
            if not dry_run:
                import shutil

                shutil.rmtree(workspace, ignore_errors=True)
                repository.delete_document(workspace.name)
    return removed
