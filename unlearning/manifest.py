"""
Writes Module 3's revision-N manifest entries into the SAME
finetuning/checkpoints/manifest.json Module 2 established (locked decision: one
checkpoints root, one manifest -- see plan.md's Open Decisions, and
finetuning/manifest.py which wrote revision-0's entry in this same file/schema).
Additive only: never renames or drops a field revision-0's entry already uses, since
app/backend (Module 5) reads this file too.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from unlearning import config as ul_config

MANIFEST_PATH = ul_config.MANIFEST_PATH


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"revisions": [], "reference_models": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def next_revision_number(manifest: Optional[dict] = None) -> int:
    manifest = manifest if manifest is not None else _load_manifest()
    existing = [e["revision"] for e in manifest["revisions"]]
    return max(existing, default=-1) + 1


def write_revision_entry(
    *,
    revision: Optional[int] = None,
    parent_revision: int,
    method: str,
    erasure_request: dict,
    adapter_path: Path,
    base_model: str,
    lora_config: dict,
    training_args: dict,
    dataset_info: dict,
    accuracy_before: dict,
    accuracy_after: dict,
    early_stop_step: Optional[int],
) -> dict:
    manifest = _load_manifest()
    revision = next_revision_number(manifest) if revision is None else revision
    entry = {
        "revision": revision,
        "label": f"{method}-{erasure_request.get('request_type')}",
        "parent_revision": parent_revision,
        "erasure_request": erasure_request,
        "method": method,
        "adapter_path": str(adapter_path),
        "base_model": base_model,
        "lora_config": lora_config,
        "training_args": training_args,
        "dataset": dataset_info,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "early_stop_step": early_stop_step,
        "created_at": _now_iso(),
    }
    manifest["revisions"] = [e for e in manifest["revisions"] if e.get("revision") != revision]
    manifest["revisions"].append(entry)
    _save_manifest(manifest)
    return entry


def read_manifest() -> dict:
    return _load_manifest()


def load_revision_adapter_path(revision: int) -> Path:
    manifest = _load_manifest()
    for e in manifest["revisions"]:
        if e["revision"] == revision:
            return Path(e["adapter_path"])
    raise KeyError(f"revision {revision} not found in {MANIFEST_PATH}")
