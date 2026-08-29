"""
Writes Module 2's revision-0 and reference-model manifest entries to
finetuning/checkpoints/manifest.json, in the shape app/backend (Module 5) is meant to
read to drive the live revision manifest (plan.md: "Module 5 ... decides how a new
request composes against prior ones" -- it needs revision-0 to already exist in a
documented shape rather than inventing the schema retroactively).

Pure Python (json/hashlib/datetime only) -- no heavy deps, so this is fully testable
without torch/transformers/peft installed.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Optional

from finetuning import config as ft_config

MANIFEST_PATH = ft_config.CHECKPOINTS_DIR / "manifest.json"


def sha256_of_file(path: Path) -> str:
    """Hash of a dataset file (e.g. data/processed/train.jsonl), recorded in every
    manifest entry so a later regeneration of Module 1's output that changes the
    data is detectable rather than silently invalidating the trained adapter's
    provenance."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"revisions": [], "reference_models": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def write_revision_0_entry(
    *,
    adapter_path: Path,
    lora_config: dict,
    training_args: dict,
    dataset_info: dict,
    eval_summary: dict,
) -> dict:
    """Registers/replaces the revision-0 entry (base LoRA adapter, trained on the
    full train set -- the starting point every unlearning run in Module 3 branches
    from, per CLAUDE.md's revision model)."""
    entry = {
        "revision": 0,
        "label": "baseline",
        "parent_revision": None,
        "erasure_request": None,
        "adapter_path": str(adapter_path),
        "base_model": ft_config.MODEL_NAME,
        "lora_config": lora_config,
        "training_args": training_args,
        "dataset": dataset_info,
        "eval_summary": eval_summary,
        "created_at": _now_iso(),
    }
    manifest = _load_manifest()
    manifest["revisions"] = [e for e in manifest["revisions"] if e.get("revision") != 0]
    manifest["revisions"].append(entry)
    _save_manifest(manifest)
    return entry


def write_reference_model_entry(
    *,
    entity: str,
    fact_group_id: str,
    adapter_path: Path,
    excluded_counts: dict,
    lora_config: dict,
    training_args: dict,
    dataset_info: dict,
    eval_summary: dict,
) -> dict:
    """Registers/replaces the retain-only reference-model entry for `entity` --
    verification ground truth (Design Doc Section 5), never a manifest "revision"
    in its own right since it's never served live or branched from."""
    entry = {
        "entity": entity,
        "fact_group_id": fact_group_id,
        "adapter_path": str(adapter_path),
        "excluded_example_counts_by_source_type": excluded_counts,
        "base_model": ft_config.MODEL_NAME,
        "lora_config": lora_config,
        "training_args": training_args,
        "dataset": dataset_info,
        "eval_summary": eval_summary,
        "created_at": _now_iso(),
    }
    manifest = _load_manifest()
    manifest["reference_models"] = [
        e for e in manifest["reference_models"] if e.get("entity") != entity
    ]
    manifest["reference_models"].append(entry)
    _save_manifest(manifest)
    return entry


def read_manifest() -> dict:
    return _load_manifest()
