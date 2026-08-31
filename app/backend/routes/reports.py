"""
GET /reports/{revision}(.md), POST /reports/{revision}/generate -- serves exactly
what verification/report.py already wrote to disk. This module never recomputes or
reshapes a signal; if a revision hasn't been verified yet, it says so explicitly
(per this repo's "never a silent omission" convention) rather than 404ing with no
explanation, and offers the on-demand generate endpoint through the same job queue
erasure requests use.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.backend import jobs, manifest_view
from verification import config as v_config

router = APIRouter(tags=["reports"])


def _report_paths(revision: int):
    json_path = v_config.REPORTS_DIR / f"revision-{revision}_verification_report.json"
    md_path = v_config.REPORTS_DIR / f"revision-{revision}_verification_report.md"
    return json_path, md_path


@router.get("/reports/{revision}")
def get_report(revision: int):
    json_path, _ = _report_paths(revision)
    if json_path.exists():
        return json.loads(json_path.read_text())

    try:
        entry = manifest_view.get_revision(revision)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"revision {revision} not found in the manifest")
    if entry.get("parent_revision") is None:
        raise HTTPException(
            status_code=400,
            detail=f"revision {revision} is the baseline (no parent_revision) -- nothing to verify it against",
        )
    raise HTTPException(
        status_code=404,
        detail=(
            f"revision {revision} exists but has not been verified yet -- "
            f"POST /reports/{revision}/generate to run verification against it"
        ),
    )


@router.get("/reports/{revision}/markdown", response_class=PlainTextResponse)
def get_report_markdown(revision: int):
    _, md_path = _report_paths(revision)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"no markdown report for revision {revision} yet")
    return md_path.read_text()


@router.post("/reports/{revision}/generate", status_code=202)
def generate_report(revision: int):
    try:
        entry = manifest_view.get_revision(revision)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"revision {revision} not found in the manifest")
    if entry.get("parent_revision") is None:
        raise HTTPException(
            status_code=400,
            detail=f"revision {revision} is the baseline -- pick a revision produced by an erasure request",
        )
    return jobs.submit_verify_only_job(revision)
