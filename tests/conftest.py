"""Shared pytest fixtures.

Tests never touch the network: SerpApi is always mocked and no real credentials
are required. A temporary data directory keeps the test run isolated from any
real documents.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.config import SerpApiSettings, Settings, load_settings  # noqa: E402


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def settings(data_dir: Path) -> Settings:
    """Settings pointing at a temporary workspace with SerpApi disabled."""
    os.environ["PAINTER_DATA_DIR"] = str(data_dir)
    base = load_settings(refresh=True)
    return Settings(
        **{
            **base.__dict__,
            "data_dir": data_dir,
            "ocr_provider": "tesseract",
            "verify_enabled": False,
            "enable_yolo": True,
            "serpapi": SerpApiSettings(api_key=None, enabled=False),
        }
    )


@pytest.fixture()
def page_image():
    """A small synthetic page image (BGR) with text and a table."""
    import cv2
    import numpy as np

    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "INVOICE 2026", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    for index in range(5):
        cv2.putText(
            image,
            f"Line {index} of body text for layout testing",
            (40, 160 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
    for row in range(1, 4):
        cv2.line(image, (40, 400 + row * 50), (560, 400 + row * 50), (0, 0, 0), 2)
    for col in range(1, 3):
        cv2.line(image, (40 + col * 173, 400), (40 + col * 173, 550), (0, 0, 0), 2)
    cv2.rectangle(image, (40, 400), (560, 550), (0, 0, 0), 2)
    return image
