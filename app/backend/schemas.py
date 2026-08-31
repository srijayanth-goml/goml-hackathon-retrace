"""
Pydantic request/response models for app/backend's routes. Kept separate from
manifest_view.py / jobs.py, which deal in plain dicts (the shapes those already
read/write -- finetuning/checkpoints/manifest.json, jobs.json) -- these are only the
HTTP-facing shapes.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    revision: int
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    revision: int
    adapter_label: str
    reply: str


class ErasureRequestBody(BaseModel):
    entity: Optional[str] = None
    attribute: Optional[str] = None
    method: Literal["npo", "ga"] = "npo"
    # None defers to unlearning/config.py's DEFAULT_PARENT_REVISION (0 -- branch
    # fresh from revision-0). Exposed as an override, per plan.md's Module 5 Open
    # Decisions, for a judge who deliberately wants to see sequential composition
    # degrade utility -- never the accidental default.
    parent_revision: Optional[int] = None
    max_steps: Optional[int] = None
    auto_verify: bool = True


class JobStatus(BaseModel):
    job_id: str
    job_type: Literal["train_and_verify", "train_only", "verify_only"]
    status: Literal["queued", "running", "verifying", "done", "failed"]
    erasure_request: Optional[dict] = None
    method: Optional[str] = None
    parent_revision: Optional[int] = None
    max_steps: Optional[int] = None
    auto_verify: bool = True
    revision: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_tail: List[str] = Field(default_factory=list)
