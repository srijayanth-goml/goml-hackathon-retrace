"""
Module 5 (App Backend) entrypoint:

    uvicorn app.backend.main:app --reload

Registers every route and opens CORS to the frontend dev server. Startup only
touches the manifest file, never imports torch -- the server boots fast and stays
usable for every route that doesn't need a real model even before torch/
transformers/peft are installed (routes/chat.py's own HeavyDepsMissing -> 503
handling covers the ones that do -- see app/backend/adapters.py).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend import config as be_config
from app.backend.routes import chat, erasure_requests, meta, reports, revisions
from unlearning import manifest as ul_manifest


@asynccontextmanager
async def lifespan(app: FastAPI):
    ul_manifest.read_manifest()  # fail fast if the manifest file itself is unreadable
    yield


app = FastAPI(
    title="ReTrace App Backend",
    description="Module 5 -- live model serving, erasure-request job queue, Erasure Report API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=be_config.CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(revisions.router)
app.include_router(chat.router)
app.include_router(erasure_requests.router)
app.include_router(reports.router)
app.include_router(meta.router)


@app.get("/health", tags=["meta"])
def health():
    manifest = ul_manifest.read_manifest()
    return {"status": "ok", "revisions_in_manifest": len(manifest["revisions"])}
