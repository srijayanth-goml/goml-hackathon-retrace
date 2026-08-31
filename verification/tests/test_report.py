"""
Tests report.py's five-section shape against a fixed, hand-built fake signal-results
dict -- independent of any real model run (report.py itself never touches a model).
"""
from verification.report import build_impact_assessment, build_report

FAKE_MANIFEST_ENTRY = {
    "revision": 1,
    "method": "npo",
    "created_at": "2026-01-01T00:00:00+00:00",
    "erasure_request": {"entity": "NeuroSync Diagnostics", "attribute": None, "request_type": "entity"},
    "parent_revision": 0,
    "training_args": {"learning_rate": 1e-4},
    "early_stop_step": 40,
}

FAKE_RESOLVED_SUMMARY = {
    "request_type": "entity",
    "entity_type": "company",
    "forget_fact_ids": ["F001", "F002", "F003", "F004", "F005"],
    "forget_fact_group_ids": ["G001"],
    "retain_neighbor_entities": ["NeuroWave Diagnostics", "NeuroCore Diagnostics"],
}

FAKE_DIRECT_QA_BEFORE = {
    "forget": {"overall_accuracy": 0.9, "n": 10, "accuracy_by_attribute": {}},
    "retain_neighbor": {"overall_accuracy": 0.85, "n": 10, "accuracy_by_attribute": {}},
    "retain_general_unrelated": {"overall_accuracy": 0.9, "n": 10, "accuracy_by_attribute": {}},
    "forget_probe": {"overall_accuracy": 0.9, "n": 4, "accuracy_by_attribute": {}},
    "forget_details": [], "forget_probe_details": [],
}
FAKE_DIRECT_QA_AFTER = {
    "forget": {"overall_accuracy": 0.05, "n": 10, "accuracy_by_attribute": {}},
    "retain_neighbor": {"overall_accuracy": 0.83, "n": 10, "accuracy_by_attribute": {}},
    "retain_general_unrelated": {"overall_accuracy": 0.89, "n": 10, "accuracy_by_attribute": {}},
    "forget_probe": {"overall_accuracy": 0.1, "n": 4, "accuracy_by_attribute": {}},
    "forget_details": [], "forget_probe_details": [],
}
FAKE_RELATIONAL = {
    "summary": {
        "n_relational_examples_probed": 2,
        "forgotten_entity_dropped_out_rate": 1.0,
        "retained_sibling_preserved_rate": 1.0,
    },
    "details": [],
}
FAKE_MIA = {
    "summary": {
        "n_forget_examples_scored": 10, "n_null_examples": 60,
        "mean_percentile_rank": 0.12, "median_percentile_rank": 0.1,
        "small_forget_set_caveat": None,
    },
    "per_example_ranks": [],
}
FAKE_REFERENCE_AVAILABLE = {
    "available": True, "reference_entity": "NeuroSync Diagnostics",
    "reference_forget_set_accuracy": {"overall_accuracy": 0.04},
}
FAKE_REFERENCE_UNAVAILABLE = {"available": False, "reason": "no reference model exists for this request"}
FAKE_GENERAL_CAPABILITY = {
    "generic_prompts": {"summary": {"n": 10, "n_correct": 10, "accuracy": 1.0}, "details": []},
    "previous_company_control_group": {
        "summary": {"overall_accuracy": 0.9, "n": 5, "accuracy_by_attribute": {}}, "details": [],
    },
}


def _build(reference_result):
    return build_report(
        manifest_entry=FAKE_MANIFEST_ENTRY,
        resolved_summary=FAKE_RESOLVED_SUMMARY,
        direct_qa_before=FAKE_DIRECT_QA_BEFORE,
        direct_qa_after=FAKE_DIRECT_QA_AFTER,
        relational_result=FAKE_RELATIONAL,
        decoy_results=[],
        mia_result=FAKE_MIA,
        reference_result=reference_result,
        general_capability_result=FAKE_GENERAL_CAPABILITY,
    )


def test_impact_assessment_flags_forget_collapse_and_neighbor_tolerance():
    impact = build_impact_assessment(FAKE_DIRECT_QA_BEFORE, FAKE_DIRECT_QA_AFTER)
    assert impact["forget_collapsed"] is True
    assert impact["neighbor_within_tolerance"] is True  # 0.85 -> 0.83, drift 0.02 <= default tolerance


def test_build_report_has_all_five_sections_non_empty():
    report = _build(FAKE_REFERENCE_AVAILABLE)
    for key in ("what_was_targeted", "what_was_done", "verification_results", "impact_assessment", "key_takeaways"):
        assert key in report and report[key], f"missing or empty section: {key}"
    assert len(report["key_takeaways"]) > 0


def test_build_report_states_unavailable_reference_explicitly_not_omitted():
    report = _build(FAKE_REFERENCE_UNAVAILABLE)
    assert report["verification_results"]["reference_model_comparison"]["available"] is False
    assert "reason" in report["verification_results"]["reference_model_comparison"]
    takeaway_text = " ".join(report["key_takeaways"])
    assert "No retain-only reference model available" in takeaway_text


def test_method_comparison_appears_only_when_provided():
    report_without = _build(FAKE_REFERENCE_AVAILABLE)
    assert report_without["verification_results"]["npo_vs_ga_comparison"] is None

    method_comparison = {
        "this_method": "npo", "this_revision": 1, "this_forget_accuracy": 0.05, "this_neighbor_accuracy": 0.83,
        "sibling_method": "ga", "sibling_revision": 2, "sibling_forget_accuracy": 0.02, "sibling_neighbor_accuracy": 0.60,
    }
    report_with = build_report(
        manifest_entry=FAKE_MANIFEST_ENTRY, resolved_summary=FAKE_RESOLVED_SUMMARY,
        direct_qa_before=FAKE_DIRECT_QA_BEFORE, direct_qa_after=FAKE_DIRECT_QA_AFTER,
        relational_result=FAKE_RELATIONAL, decoy_results=[], mia_result=FAKE_MIA,
        reference_result=FAKE_REFERENCE_AVAILABLE, general_capability_result=FAKE_GENERAL_CAPABILITY,
        method_comparison=method_comparison,
    )
    assert report_with["verification_results"]["npo_vs_ga_comparison"] == method_comparison
    assert any("NPO vs GA" in b for b in report_with["key_takeaways"])
