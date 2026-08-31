"""GET /revisions, GET /revisions/{revision} -- the normalized manifest view."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend import manifest_view

router = APIRouter(tags=["revisions"])


@router.get("/revisions")
def list_revisions():
    return manifest_view.list_revisions()


@router.get("/revisions/{revision}")
def get_revision(revision: int):
    try:
        return manifest_view.get_revision(revision)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"revision {revision} not found in the manifest")
