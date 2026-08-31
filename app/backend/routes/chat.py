"""
POST /chat -- live chat against any manifest revision. This is the concrete proof of
Design Doc Section 8's "compare pre- and post-erasure live, not a recorded demo":
the same request shape against revision=0 and revision=N returns each adapter's own
actual current behavior.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.backend import config as be_config, manifest_view
from app.backend.adapters import HeavyDepsMissing, get_cache
from app.backend.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        manifest_view.get_revision(body.revision)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"revision {body.revision} not found in the manifest")

    max_new_tokens = min(
        body.max_new_tokens or be_config.DEFAULT_MAX_NEW_TOKENS, be_config.MAX_NEW_TOKENS_CAP
    )
    messages = [m.model_dump() for m in body.messages]

    cache = get_cache()
    try:
        reply = cache.generate(body.revision, messages, max_new_tokens)
    except HeavyDepsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(revision=body.revision, adapter_label=cache.adapter_label(body.revision), reply=reply)
