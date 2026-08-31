"""
FastAPI TestClient smoke tests across every route. No torch/transformers/peft
needed for any of these -- the one route that would need them (POST /chat against a
real revision) is tested only for its 404-before-touching-the-model path here;
app/backend/tests/test_request_validation.py covers what happens when a REAL job
reaches the model-dependent code, which is where a missing-heavy-deps environment
would actually surface.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_revisions_includes_baseline():
    r = client.get("/revisions")
    assert r.status_code == 200
    assert any(rev["revision"] == 0 for rev in r.json())


def test_get_unknown_revision_404():
    r = client.get("/revisions/9999")
    assert r.status_code == 404


def test_list_entities_covers_all_100():
    r = client.get("/entities")
    assert r.status_code == 200
    entities = r.json()
    assert len(entities) == 100
    assert any(e["entity"] == "NeuroSync Diagnostics" and e["entity_type"] == "company" for e in entities)


def test_list_attributes():
    r = client.get("/attributes")
    assert r.status_code == 200
    body = r.json()
    assert "ceo" in body["company"]
    assert "education" in body["person"]


def test_list_example_requests_skips_deprecated():
    r = client.get("/requests/examples")
    assert r.status_code == 200
    names = [e["name"] for e in r.json()]
    assert "silvergate_aerospace_entity" not in names
    assert "neurosync_entity" in names


def test_report_for_baseline_revision_400():
    r = client.get("/reports/0")
    assert r.status_code == 400


def test_report_for_unknown_revision_404():
    r = client.get("/reports/9999")
    assert r.status_code == 404


def test_generate_report_for_unknown_revision_404():
    r = client.post("/reports/9999/generate")
    assert r.status_code == 404


def test_chat_against_unknown_revision_404():
    r = client.post("/chat", json={"revision": 9999, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


def test_chat_rejects_malformed_message_role():
    r = client.post("/chat", json={"revision": 0, "messages": [{"role": "narrator", "content": "hi"}]})
    assert r.status_code == 422  # pydantic validation, not a route bug


def test_jobs_list_empty_or_well_shaped():
    r = client.get("/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
