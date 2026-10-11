"""Painter — AI document reconstruction and replication engine.

Painter reconstructs an editable document (HTML / PDF / DOCX) from a scanned or
photographed page. The public entry points are:

* :func:`painter.pipeline.process_document` — run the full pipeline on a file.
* :mod:`painter.api` — FastAPI application used by the web frontend.
* :mod:`painter.cli` — command line interface (``python -m painter.cli``).

The legacy single-file demo lives in ``yolo_demo/`` and now reuses these modules
instead of duplicating detection and recognition logic.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
