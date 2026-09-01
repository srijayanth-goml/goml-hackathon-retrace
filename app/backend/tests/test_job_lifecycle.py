"""
jobs.py's state machine, exercised directly via _run_job (bypassing the queue/worker
thread for a deterministic test) with FAKE _do_training/_do_verification -- no real
model needed. Jobs are registered with jobs._register (NOT submit_training_job/
submit_verify_only_job) specifically so the real background worker thread never also
picks them up and races the test's own direct _run_job call -- submit_*'s queueing
behavior is exercised separately, for real, by test_request_validation.py's last
test (no mocking there, against whatever torch/transformers/peft actually are
installed in this environment).
"""
from __future__ import annotations

import datetime
import json
import uuid

from app.backend import jobs


class _FakeCache:
    def __init__(self):
        self.refreshed = False

    def refresh(self):
        self.refreshed = True
        return []


def _patch_jobs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs.be_config, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(jobs.be_config, "JOBS_JSON_PATH", tmp_path / "jobs.json")


def _make_training_job(auto_verify: bool) -> dict:
    job = {
        "job_id": str(uuid.uuid4()),
        "job_type": "train_and_verify" if auto_verify else "train_only",
        "status": "queued",
        "erasure_request": {"entity": "NeuroSync Diagnostics", "attribute": None, "request_type": "entity"},
        "method": "npo",
        "parent_revision": None,
        "max_steps": None,
        "auto_verify": auto_verify,
        "revision": None,
        "error": None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "log_tail": [],
    }
    jobs._register(job)
    return job


def _make_verify_only_job(revision: int) -> dict:
    job = {
        "job_id": str(uuid.uuid4()),
        "job_type": "verify_only",
        "status": "queued",
        "erasure_request": None,
        "method": None,
        "parent_revision": None,
        "max_steps": None,
        "auto_verify": True,
        "revision": revision,
        "error": None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "log_tail": [],
    }
    jobs._register(job)
    return job


def test_train_and_verify_job_success(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(jobs, "_do_training", lambda job: {"revision": 7})
    monkeypatch.setattr(jobs, "_do_verification", lambda revision: calls.append(revision))
    fake_cache = _FakeCache()
    monkeypatch.setattr(jobs, "get_cache", lambda: fake_cache)

    job = _make_training_job(auto_verify=True)
    jobs._run_job(job["job_id"])

    result = jobs.get_job(job["job_id"])
    assert result["status"] == "done"
    assert result["revision"] == 7
    assert calls == [7]
    assert fake_cache.refreshed is True


def test_train_only_job_skips_verification(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)
    verify_calls = []
    monkeypatch.setattr(jobs, "_do_training", lambda job: {"revision": 3})
    monkeypatch.setattr(jobs, "_do_verification", lambda revision: verify_calls.append(revision))
    monkeypatch.setattr(jobs, "get_cache", lambda: _FakeCache())

    job = _make_training_job(auto_verify=False)
    jobs._run_job(job["job_id"])

    assert jobs.get_job(job["job_id"])["status"] == "done"
    assert verify_calls == []


def test_training_failure_records_error_and_log_tail(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)

    def boom(job):
        print("about to explode")
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(jobs, "_do_training", boom)

    job = _make_training_job(auto_verify=False)
    jobs._run_job(job["job_id"])

    result = jobs.get_job(job["job_id"])
    assert result["status"] == "failed"
    assert "simulated training failure" in result["error"]
    assert "about to explode" in result["log_tail"]


def test_system_exit_from_missing_heavy_deps_fails_the_job_not_the_process(monkeypatch, tmp_path):
    # Regression test: unlearning/train.py's _require_heavy_deps() raises SystemExit
    # (a BaseException, not an Exception) when torch/transformers/peft aren't
    # installed -- a plain `except Exception` around this would let it escape and
    # kill the worker thread instead of just failing the one job. Caught for real by
    # this test before the fix.
    #
    # _patch_jobs_dir was missing here for a while -- every other test in this file
    # calls it, but this one didn't, so running this suite wrote a real
    # "simulated missing heavy deps" job straight into app/backend/jobs/jobs.json
    # (the actual runtime file, not a tmp_path). Caught by finding exactly that
    # string in a real jobs.json outside any test run.
    _patch_jobs_dir(monkeypatch, tmp_path)

    def raises_system_exit(job):
        raise SystemExit("simulated missing heavy deps")

    monkeypatch.setattr(jobs, "_do_training", raises_system_exit)

    job = _make_training_job(auto_verify=False)
    jobs._run_job(job["job_id"])  # must not propagate SystemExit out of this call

    result = jobs.get_job(job["job_id"])
    assert result["status"] == "failed"
    assert "simulated missing heavy deps" in result["error"]


def test_verify_only_job_calls_do_verification_with_its_revision(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(jobs, "_do_verification", lambda revision: calls.append(revision))
    monkeypatch.setattr(jobs, "get_cache", lambda: _FakeCache())

    job = _make_verify_only_job(revision=4)
    jobs._run_job(job["job_id"])

    assert calls == [4]
    result = jobs.get_job(job["job_id"])
    assert result["status"] == "done"
    assert result["revision"] == 4


def test_job_history_persists_to_disk(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(jobs, "_do_training", lambda job: {"revision": 1})
    monkeypatch.setattr(jobs, "_do_verification", lambda revision: None)
    monkeypatch.setattr(jobs, "get_cache", lambda: _FakeCache())

    job = _make_training_job(auto_verify=True)
    jobs._run_job(job["job_id"])

    assert jobs.be_config.JOBS_JSON_PATH.exists()
    import json
    on_disk = json.loads(jobs.be_config.JOBS_JSON_PATH.read_text())
    assert any(j["job_id"] == job["job_id"] and j["status"] == "done" for j in on_disk)


def test_submit_training_job_returns_an_independent_snapshot(monkeypatch, tmp_path):
    # Regression test: submit_training_job used to return the SAME mutable dict
    # object stored in jobs._jobs -- if the background worker mutates it (e.g. to
    # "running") before the caller reads the returned value, the HTTP response could
    # show a status that was already stale. submit_* must return a copy.
    _patch_jobs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(jobs, "_do_training", lambda job: {"revision": 1})
    monkeypatch.setattr(jobs, "_do_verification", lambda revision: None)
    monkeypatch.setattr(jobs, "get_cache", lambda: _FakeCache())

    from unlearning.request import ErasureRequest
    snapshot = jobs.submit_training_job(
        ErasureRequest(entity="NeuroSync Diagnostics"), method="npo",
        parent_revision=None, max_steps=None, auto_verify=False,
    )
    assert snapshot["status"] == "queued"

    # let the real worker thread actually finish the job
    import time
    deadline = time.time() + 5
    while time.time() < deadline and jobs.get_job(snapshot["job_id"])["status"] in ("queued", "running"):
        time.sleep(0.05)

    # the snapshot returned at submission time must be untouched by the later mutation
    assert snapshot["status"] == "queued"
    assert jobs.get_job(snapshot["job_id"])["status"] == "done"


def test_load_persisted_reconciles_stale_running_jobs_on_startup(monkeypatch, tmp_path):
    # Regression test for the real bug this class of scenario caused: a job
    # persisted as "running" by a process that no longer exists (a crash, a
    # Ctrl+C, an interrupted `pytest` run) must NOT stay "running" forever just
    # because a new process loaded it from disk -- this process's worker thread
    # always starts empty, so nothing could actually still be running it. Before
    # the fix, _load_persisted() loaded such a record verbatim, and
    # useActiveJob()-style "is anything active" checks (the real frontend's
    # ErasureRequestForm included) would block every future submission forever.
    _patch_jobs_dir(monkeypatch, tmp_path)

    stale_job_id = str(uuid.uuid4())
    stale_record = {
        "job_id": stale_job_id,
        "job_type": "train_and_verify",
        "status": "running",
        "erasure_request": {"entity": "NeuroSync Diagnostics", "attribute": None, "request_type": "entity"},
        "method": "npo",
        "parent_revision": None,
        "max_steps": None,
        "auto_verify": True,
        "revision": None,
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": None,
        "log_tail": [],
    }
    jobs.be_config.JOBS_JSON_PATH.write_text(json.dumps([stale_record]))

    jobs._load_persisted()

    result = jobs.get_job(stale_job_id)
    assert result["status"] == "failed"
    assert "restart" in result["error"] or "no longer running" in result["error"]
    assert result["finished_at"] is not None

    # The reconciliation must have been written back to disk too, not just held
    # in memory -- a second process loading this same file should see "failed",
    # not "running" again.
    on_disk = json.loads(jobs.be_config.JOBS_JSON_PATH.read_text())
    on_disk_record = next(r for r in on_disk if r["job_id"] == stale_job_id)
    assert on_disk_record["status"] == "failed"


def test_load_persisted_leaves_terminal_status_jobs_untouched(monkeypatch, tmp_path):
    _patch_jobs_dir(monkeypatch, tmp_path)

    done_job_id = str(uuid.uuid4())
    done_record = {
        "job_id": done_job_id,
        "job_type": "verify_only",
        "status": "done",
        "erasure_request": None,
        "method": None,
        "parent_revision": None,
        "max_steps": None,
        "auto_verify": True,
        "revision": 1,
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
        "log_tail": [],
    }
    jobs.be_config.JOBS_JSON_PATH.write_text(json.dumps([done_record]))

    jobs._load_persisted()

    result = jobs.get_job(done_job_id)
    assert result["status"] == "done"
    assert result["error"] is None
    assert result["finished_at"] == "2026-01-01T00:00:02+00:00"
