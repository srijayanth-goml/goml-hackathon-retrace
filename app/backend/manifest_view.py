"""
Normalizes finetuning/checkpoints/manifest.json's revision entries into ONE shared
shape for the API/UI. Revision-0's entry (finetuning/manifest.py) and revision-N
entries (unlearning/manifest.py) are NOT actually the same schema despite
unlearning/manifest.py's own docstring calling it "the SAME schema" -- revision-0
carries `eval_summary` (a heldout sanity-check accuracy, unrelated to any erasure
request), revision-N carries `accuracy_before`/`accuracy_after`/`early_stop_step`
(forget/neighbor/general/forget_probe accuracy for THAT specific request), and
neither has the other's fields. Forcing both into one plain "accuracy: 0.42" number
would conflate two different measurements -- instead this exposes both raw under an
explicit "kind" discriminator plus a best-effort `headline_accuracy` for simple
display, the same honesty posture verification/report.py takes with its own signals
(never silently omitting or conflating).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from unlearning import manifest as ul_manifest
from verification import config as v_config


def _report_path(revision: int) -> Path:
    return v_config.REPORTS_DIR / f"revision-{revision}_verification_report.json"


def has_verification_report(revision: int) -> bool:
    return _report_path(revision).exists()


def _normalize_accuracy(entry: dict) -> dict:
    if "eval_summary" in entry:
        eval_summary = entry["eval_summary"]
        return {
            "kind": "baseline_eval_summary",
            "eval_summary": eval_summary,
            # the heldout sanity-check number (finetuning/eval_quick.py) -- NOT a
            # forget-accuracy number, since revision-0 has no erasure request
            "headline_accuracy": eval_summary.get("overall_accuracy"),
        }
    accuracy_before = entry.get("accuracy_before")
    accuracy_after = entry.get("accuracy_after")
    headline = None
    if accuracy_after and accuracy_after.get("forget"):
        headline = accuracy_after["forget"].get("overall_accuracy")
    return {
        "kind": "unlearning_accuracy_before_after",
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "early_stop_step": entry.get("early_stop_step"),
        # the "did it forget" number -- forget-set accuracy AFTER unlearning.
        # Module 3's own self-reported number (a useful cross-check per plan.md's
        # Module 4, never a substitute for verification/reports/*'s independently
        # recomputed one).
        "headline_accuracy": headline,
    }


def _normalize_entry(entry: dict) -> dict:
    revision = entry["revision"]
    return {
        "revision": revision,
        "label": entry.get("label"),
        "parent_revision": entry.get("parent_revision"),
        "method": entry.get("method"),
        "erasure_request": entry.get("erasure_request"),
        "base_model": entry.get("base_model"),
        "adapter_path": entry.get("adapter_path"),
        "lora_config": entry.get("lora_config"),
        "training_args": entry.get("training_args"),
        "created_at": entry.get("created_at"),
        "accuracy": _normalize_accuracy(entry),
        "has_verification_report": has_verification_report(revision),
    }


def list_revisions() -> List[dict]:
    manifest = ul_manifest.read_manifest()
    return sorted((_normalize_entry(e) for e in manifest["revisions"]), key=lambda r: r["revision"])


def get_revision(revision: int) -> dict:
    manifest = ul_manifest.read_manifest()
    for entry in manifest["revisions"]:
        if entry["revision"] == revision:
            return _normalize_entry(entry)
    raise KeyError(f"revision {revision} not found in {ul_manifest.MANIFEST_PATH}")
