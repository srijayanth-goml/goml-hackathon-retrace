"""
POST /erasure-requests validates synchronously, torch-free, BEFORE enqueueing a job
-- this is the check that already caught the heldout-entity bug during Module 3's
own testing (see unlearning/requests/silvergate_aerospace_entity.json's own
deprecation note), now exercised at the API boundary against the REAL dataset.

The last test in this file submits a genuinely VALID request and follows it through
the real background queue -- no mocking -- to prove a valid submission enqueues and
the worker thread actually picks it up. Whatever happens after that (a real reply,
or a clean "torch not installed" failure) depends on this environment's installed
packages, which is exactly the behavior app/backend/adapters.py is supposed to
guarantee either way: never a silent hang, never a raw traceback.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_rejects_empty_request():
    r = client.post("/erasure-requests", json={})
    assert r.status_code == 400


def test_rejects_unknown_entity():
    r = client.post("/erasure-requests", json={"entity": "Not A Real Company"})
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_rejects_heldout_entity_with_nothing_genuine_to_unlearn():
    # Silvergate Aerospace is a REAL entity in the fact table but landed in Module
    # 1's heldout split -- unlearning/data.py's own guard, exercised here through
    # the API. See unlearning/requests/silvergate_aerospace_entity.json.
    r = client.post("/erasure-requests", json={"entity": "Silvergate Aerospace"})
    assert r.status_code == 400
    assert "heldout" in r.json()["detail"].lower()


def test_valid_request_is_accepted_and_reaches_the_worker():
    r = client.post("/erasure-requests", json={"entity": "NeuroSync Diagnostics", "attribute": "ceo"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # The background worker can start processing before this response is even
    # serialized, so "queued" isn't guaranteed by the time we read it here --
    # submit_training_job returning an independent SNAPSHOT (not the live mutable
    # dict) is what's actually being guarded: whatever status comes back must be
    # one the job legitimately passed through, not a torn/inconsistent read.
    assert r.json()["status"] in ("queued", "running", "verifying", "done", "failed")

    # Give the background worker a moment to pick it up. _require_heavy_deps() fails
    # near-instantly if torch/transformers/peft aren't installed, so this should
    # resolve quickly either way -- poll rather than sleep-once, to avoid flakiness.
    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status != "queued" and status != "running" and status != "verifying":
            break
        time.sleep(0.1)

    assert status in ("done", "failed")
    if status == "failed":
        # the only expected failure reason in an environment without the heavy deps
        detail = client.get(f"/jobs/{job_id}").json()["error"]
        assert "torch" in detail or "transformers" in detail or "peft" in detail
