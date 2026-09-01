"""
Background job queue for POST /erasure-requests and POST /reports/{revision}/generate.
ONE worker thread (not a pool) -- jobs never run concurrently with each other, and
_run_job takes adapters.MODEL_LOCK for its whole duration so a chat request never
runs a forward pass at the same moment as a training/verification job's forward+
backward passes. See plan.md's Module 5 "hardware fact" for why this is a hard
constraint on the laptop this runs on, not a style choice.

Persists job history to app/backend/jobs/jobs.json (flat file, not a database --
matches every other module's report-next-to-its-output convention) so a server
restart mid-demo doesn't lose the record of what ran.

`_do_training`/`_do_verification` are the two points where this module reaches out
to unlearning.train.run / verification.run_verification.run -- kept as separate,
patchable functions specifically so tests can substitute a fake job body without
needing torch/transformers/peft installed (see tests/test_job_lifecycle.py).
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import queue
import threading
import uuid
from typing import Dict, List, Optional

from app.backend import config as be_config
from app.backend.adapters import MODEL_LOCK, get_cache
from unlearning.request import ErasureRequest

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()   # protects the in-memory _jobs dict + jobs.json write --
                                  # distinct from MODEL_LOCK, which protects the model itself
_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _persist() -> None:
    with _jobs_lock:
        snapshot = list(_jobs.values())
    be_config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    be_config.JOBS_JSON_PATH.write_text(json.dumps(snapshot, indent=2))


_STALE_STATUSES = ("queued", "running", "verifying")


def _load_persisted() -> None:
    """Loads jobs.json into memory at import time, then reconciles anything left in
    a non-terminal status (queued/running/verifying). This process's `_worker_loop`
    ALWAYS starts empty -- there is no way for a job persisted as "running" to
    actually still be running in a freshly-started process, since the only worker
    that could have been running it lived in whatever process wrote that record and
    is gone now (a killed/crashed server, a Ctrl+C'd `pytest` run, etc.). Without
    this, a stale "running" record blocks every future submission forever (see
    plan.md's Module 6 notes on this -- `useActiveJob()` correctly treats any
    active-status job as blocking, so an un-reconciled stale one reads as "a job is
    already running" when nothing actually is)."""
    if not be_config.JOBS_JSON_PATH.exists():
        return
    try:
        records = json.loads(be_config.JOBS_JSON_PATH.read_text())
    except json.JSONDecodeError:
        return  # corrupt job history is not worth crashing startup over
    reconciled = False
    for r in records:
        stale_status = r.get("status")
        if stale_status in _STALE_STATUSES:
            r["status"] = "failed"
            r["error"] = (
                "no worker thread carried this job over a server restart -- it was "
                f"left as {stale_status!r} by a process that is no longer running "
                "(a crash, a Ctrl+C, or an interrupted test run). Submit the request "
                "again if it still needs to run."
            )
            r["finished_at"] = _now_iso()
            reconciled = True
        _jobs[r["job_id"]] = r
    if reconciled:
        _persist()


def _update(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)
    _persist()


def list_jobs() -> List[dict]:
    with _jobs_lock:
        return sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)


def get_job(job_id: str) -> dict:
    with _jobs_lock:
        if job_id not in _jobs:
            raise KeyError(job_id)
        return dict(_jobs[job_id])


def _register(job: dict) -> None:
    """Stores a freshly-built job record (does NOT enqueue it -- callers do that
    separately). Split out from submit_training_job/submit_verify_only_job so tests
    can build a job dict and register it without the real background worker racing
    to process it at the same time a test also calls _run_job directly (see
    tests/test_job_lifecycle.py)."""
    with _jobs_lock:
        _jobs[job["job_id"]] = job
    _persist()


def submit_training_job(
    request: ErasureRequest,
    method: str,
    parent_revision: Optional[int],
    max_steps: Optional[int],
    auto_verify: bool,
) -> dict:
    """Caller (routes/erasure_requests.py) must already have validated `request`
    synchronously -- via unlearning.data.build_unlearning_batches, torch-free -- BEFORE
    calling this. This function never re-validates, it only queues the heavy run.
    Returns a SNAPSHOT (dict copy) of the job at submission time -- the background
    worker can start mutating the live record immediately after _queue.put, so the
    HTTP response must not be the same mutable object."""
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "job_type": "train_and_verify" if auto_verify else "train_only",
        "status": "queued",
        "erasure_request": request.to_dict(),
        "method": method,
        "parent_revision": parent_revision,
        "max_steps": max_steps,
        "auto_verify": auto_verify,
        "revision": None,
        "error": None,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "log_tail": [],
    }
    _register(job)
    _ensure_worker_started()
    _queue.put(job_id)
    return dict(job)


def submit_verify_only_job(revision: int) -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "job_type": "verify_only",
        "status": "queued",
        "erasure_request": None,
        "method": None,
        "parent_revision": None,
        "max_steps": None,
        "auto_verify": True,
        "revision": revision,
        "error": None,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "log_tail": [],
    }
    _register(job)
    _ensure_worker_started()
    _queue.put(job_id)
    return dict(job)


def _ensure_worker_started() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    t = threading.Thread(target=_worker_loop, name="retrace-job-worker", daemon=True)
    t.start()


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- a job must never take the
            # worker thread down. SystemExit specifically: unlearning/train.py and
            # verification/run_verification.py's own _require_heavy_deps() raise
            # SystemExit (matching their CLI posture), which is a BaseException, not
            # an Exception -- a plain `except Exception` would let it silently kill
            # this whole worker thread instead of just failing the one job.
            _update(job_id, status="failed", error=str(exc), finished_at=_now_iso())


def _do_training(job: dict) -> dict:
    from unlearning import train as unlearning_train
    request = ErasureRequest.from_dict(job["erasure_request"])
    return unlearning_train.run(
        request,
        method=job["method"],
        parent_revision=job["parent_revision"],
        max_steps=job["max_steps"],
    )


def _do_verification(revision: int) -> None:
    from verification import run_verification
    run_verification.run(revision)


def _run_job(job_id: str) -> None:
    """Runs one job to completion. Bypasses the queue -- called by _worker_loop for
    a real job, and directly by tests for a deterministic, synchronous check of the
    state machine (see tests/test_job_lifecycle.py)."""
    job = get_job(job_id)
    _update(job_id, status="running", started_at=_now_iso())
    buf = io.StringIO()
    revision = job.get("revision")  # already known for verify_only jobs

    with MODEL_LOCK:  # never overlap a job with a chat generation call -- see module docstring
        try:
            with contextlib.redirect_stdout(buf):
                if job["job_type"] == "verify_only":
                    _do_verification(revision)
                else:
                    entry = _do_training(job)
                    revision = entry["revision"]
                    _update(job_id, revision=revision)
                    if job["auto_verify"]:
                        _update(job_id, status="verifying")
                        _do_verification(revision)
            get_cache().refresh()
        except (Exception, SystemExit) as exc:
            # SystemExit: see _worker_loop's own comment -- _require_heavy_deps()
            # raises it, and it must fail just this job, not the whole thread/process.
            log_tail = buf.getvalue().splitlines()[-be_config.LOG_TAIL_MAX_LINES:]
            _update(job_id, status="failed", error=str(exc), finished_at=_now_iso(), log_tail=log_tail)
            return

    log_tail = buf.getvalue().splitlines()[-be_config.LOG_TAIL_MAX_LINES:]
    _update(job_id, status="done", revision=revision, finished_at=_now_iso(), log_tail=log_tail)


_load_persisted()
