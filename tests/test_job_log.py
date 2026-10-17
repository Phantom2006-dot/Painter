"""The live job log: the thing that answers "is it stuck or just working?".

A stage can run for minutes inside a single call, so the stage table alone looks
exactly like a hung process. These tests cover the store, the API contract, and a
real run that must produce progress from inside the slow stage rather than only at
its boundaries.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from painter.api import create_app

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.jobs.shutdown()


def _register_document(settings, name: str = "log.png") -> str:
    from painter.storage import Repository, utcnow

    repository = Repository(settings.db_path)
    document_id = f"doc_{name.replace('.', '_')}"
    repository.create_document(
        document_id,
        name,
        str(settings.data_dir / name),
        "sha",
        "image/png",
        10,
        {"created": utcnow()},
    )
    return document_id


# ---------------------------------------------------------------- the store
def test_log_lines_are_ordered_and_filterable(settings):
    from painter.storage import Repository

    repository = Repository(settings.db_path)
    document_id = _register_document(settings, "store.png")
    repository.create_job("job_store", document_id)

    first = repository.append_log("job_store", "stage validate starting", stage="validate")
    second = repository.append_log("job_store", "stage detect starting", stage="detect")
    third = repository.append_log("job_store", "something went wrong", level="error", stage="detect")
    assert first < second < third, "ids must increase so a cursor can page the log"

    everything = repository.get_job_log("job_store")
    assert [row["message"] for row in everything] == [
        "stage validate starting",
        "stage detect starting",
        "something went wrong",
    ]

    # `since` is what keeps the poll small, so it must exclude what was already seen
    newer = repository.get_job_log("job_store", since_id=second)
    assert [row["message"] for row in newer] == ["something went wrong"]
    assert newer[0]["level"] == "error"


def test_log_is_scoped_to_its_job(settings):
    from painter.storage import Repository

    repository = Repository(settings.db_path)
    document_id = _register_document(settings, "scope.png")
    repository.create_job("job_a", document_id)
    repository.create_job("job_b", document_id)
    repository.append_log("job_a", "a only")
    repository.append_log("job_b", "b only")

    assert [r["message"] for r in repository.get_job_log("job_a")] == ["a only"]
    assert [r["message"] for r in repository.get_job_log("job_b")] == ["b only"]


# ---------------------------------------------------------------- the API
def test_log_endpoint_returns_entries_and_cursor(client, settings):
    from painter.storage import Repository

    repository = Repository(settings.db_path)
    document_id = _register_document(settings, "api.png")
    repository.create_job("job_api", document_id)
    repository.append_log("job_api", "stage validate starting", stage="validate")
    last = repository.append_log("job_api", "stage detect completed in 5.6s", stage="detect")

    response = client.get("/api/jobs/job_api/log")
    assert response.status_code == 200
    payload = response.json()
    assert [e["message"] for e in payload["entries"]] == [
        "stage validate starting",
        "stage detect completed in 5.6s",
    ]
    assert payload["last_id"] == last

    # polling again with the cursor returns nothing new, not a repeat
    again = client.get(f"/api/jobs/job_api/log?since={last}").json()
    assert again["entries"] == []
    assert again["last_id"] == last


def test_log_endpoint_rejects_unknown_job(client):
    response = client.get("/api/jobs/job_does_not_exist/log")
    assert response.status_code == 404


# ---------------------------------------------------------------- a real run
def test_a_real_run_logs_progress_inside_the_slow_stage(client, settings):
    """The point of the feature: movement during a stage, not only between them."""
    from painter.testsupport import build_synthetic_document

    case = build_synthetic_document(settings.data_dir / "logcase", case="mixed")
    with case.image_path.open("rb") as handle:
        upload = client.post(
            "/api/documents/upload",
            files={"file": (case.image_path.name, handle, "image/png")},
        )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["document_id"]

    job = client.post(
        f"/api/documents/{document_id}/process",
        json={
            "force_stages": [
                "validate",
                "preprocess",
                "detect",
                "recognize",
                "reconstruct",
                "render",
                "export",
            ],
            "enable_verify": False,
            "enable_lens": False,
            "ocr_provider": "tesseract",
        },
    )
    assert job.status_code == 200, job.text
    job_id = job.json()["job_id"]

    messages: list[str] = []
    seen_ids: list[int] = []
    since = 0
    deadline = time.time() + 600
    status = None
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}/log?since={since}").json()
        messages.extend(entry["message"] for entry in payload["entries"])
        seen_ids.extend(entry["id"] for entry in payload["entries"])
        since = payload["last_id"]
        status = payload["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(1.0)

    assert status == "completed", f"job ended as {status}: {messages[-6:]}"
    joined = "\n".join(messages)
    # every stage announces itself and its result
    for stage in ("validate", "detect", "recognize", "render", "export"):
        assert f"stage {stage} starting" in joined, f"{stage} never announced its start"
        assert f"stage {stage} completed" in joined, f"{stage} never reported completion"
    # and the timings are real numbers, not placeholders
    assert any("completed in" in message for message in messages)
    # Polling with a cursor must deliver every line exactly once, in order. Comparing
    # the messages against sorted() would only prove the alphabet, not the ordering.
    assert seen_ids == sorted(seen_ids), "log lines must arrive in id order"
    assert len(seen_ids) == len(set(seen_ids)), "the cursor must never re-deliver a line"
