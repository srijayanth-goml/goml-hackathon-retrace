"""
POST /erasure-requests -- submit a new erasure request and kick off Module 3 (+
Module 4, if auto_verify) against it as a background job. GET /jobs, GET /jobs/{id}
-- job history/status.

Validation happens HERE, synchronously, torch-free: unlearning.data.
build_unlearning_batches(request) is the exact same resolution unlearning.train.run
will do anyway, so calling it before enqueueing catches a bad request (unknown
entity, a heldout entity with nothing genuine to unlearn -- unlearning/data.py's own
real bug catch) in the HTTP response, not three minutes into a background job.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend import jobs
from app.backend.schemas import ErasureRequestBody
from unlearning.data import build_unlearning_batches
from unlearning.request import ErasureRequest

router = APIRouter(tags=["erasure-requests"])


@router.post("/erasure-requests", status_code=202)
def submit_erasure_request(body: ErasureRequestBody):
    try:
        request = ErasureRequest(entity=body.entity, attribute=body.attribute)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        build_unlearning_batches(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"request does not resolve to anything genuinely unlearnable: {exc}",
        ) from exc

    return jobs.submit_training_job(
        request,
        method=body.method,
        parent_revision=body.parent_revision,
        max_steps=body.max_steps,
        auto_verify=body.auto_verify,
    )


@router.get("/jobs")
def list_jobs():
    return jobs.list_jobs()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
