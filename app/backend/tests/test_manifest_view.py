"""
manifest_view.py's normalization, run against the REAL manifest.json for the
revision-0 shape (the only real revision that exists so far), plus a hand-built
fixture for the revision-N (accuracy_before/after) shape that doesn't exist in the
real manifest yet -- same posture verification/tests/test_report.py takes for
shapes that need a real training run to exist for real.
"""
from __future__ import annotations

import pytest

from app.backend import manifest_view


def test_revision_0_normalizes_with_baseline_kind():
    entry = manifest_view.get_revision(0)
    assert entry["revision"] == 0
    assert entry["parent_revision"] is None
    assert entry["accuracy"]["kind"] == "baseline_eval_summary"
    assert "eval_summary" in entry["accuracy"]
    assert entry["accuracy"]["headline_accuracy"] == entry["accuracy"]["eval_summary"]["overall_accuracy"]


def test_revision_0_has_no_verification_report():
    entry = manifest_view.get_revision(0)
    assert entry["has_verification_report"] is False


def test_list_revisions_includes_revision_0_sorted():
    revisions = manifest_view.list_revisions()
    assert revisions[0]["revision"] == 0
    assert all(revisions[i]["revision"] <= revisions[i + 1]["revision"] for i in range(len(revisions) - 1))


def test_unknown_revision_raises_keyerror():
    with pytest.raises(KeyError):
        manifest_view.get_revision(9999)


def test_unlearning_shaped_entry_normalizes_with_unlearning_kind():
    # A revision-N entry, hand-built to unlearning/manifest.py's real shape (no real
    # one exists yet -- Module 3 hasn't produced revision-1 for real). Exercises the
    # OTHER branch of _normalize_accuracy without needing a real training run.
    fake_entry = {
        "revision": 1,
        "label": "npo-entity",
        "parent_revision": 0,
        "erasure_request": {"entity": "NeuroSync Diagnostics", "attribute": None, "request_type": "entity"},
        "method": "npo",
        "adapter_path": "finetuning/checkpoints/revision-1-npo",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_config": {},
        "training_args": {},
        "dataset": {},
        "accuracy_before": {"forget": {"overall_accuracy": 0.95}, "neighbor": {"overall_accuracy": 0.9}},
        "accuracy_after": {"forget": {"overall_accuracy": 0.02}, "neighbor": {"overall_accuracy": 0.88}},
        "early_stop_step": 40,
        "created_at": "2026-08-31T00:00:00+00:00",
    }
    normalized = manifest_view._normalize_entry(fake_entry)
    assert normalized["accuracy"]["kind"] == "unlearning_accuracy_before_after"
    assert normalized["accuracy"]["headline_accuracy"] == 0.02
    assert normalized["accuracy"]["early_stop_step"] == 40
    assert "eval_summary" not in normalized["accuracy"]
