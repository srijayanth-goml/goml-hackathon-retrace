"""
Assembles the Erasure Report (Design Doc Section 7 / the brief's own flow diagram):
What Was Targeted, What Was Done, Verification Results, Impact Assessment, Key
Takeaways -- generated automatically from a manifest revision entry plus every
signal from direct_qa.py/relational_probe.py/mia.py/reference_comparison.py/
general_capability.py, never hand-written, so it stays honest and reproducible
(CLAUDE.md's provability convention). Writes .json (for app/backend to render) and
.md (for a human/judge reading it directly), same convention as every other
module's reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from verification import config as v_config


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _drift(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return before - after


def build_impact_assessment(direct_qa_before: dict, direct_qa_after: dict) -> dict:
    neighbor_before = direct_qa_before["retain_neighbor"]["overall_accuracy"]
    neighbor_after = direct_qa_after["retain_neighbor"]["overall_accuracy"]
    general_before = direct_qa_before["retain_general_unrelated"]["overall_accuracy"]
    general_after = direct_qa_after["retain_general_unrelated"]["overall_accuracy"]

    neighbor_drift = _drift(neighbor_before, neighbor_after)
    general_drift = _drift(general_before, general_after)
    forget_after = direct_qa_after["forget"]["overall_accuracy"]

    return {
        "neighbor_accuracy_before": neighbor_before,
        "neighbor_accuracy_after": neighbor_after,
        "neighbor_drift": neighbor_drift,
        "neighbor_within_tolerance": (
            neighbor_drift is not None and neighbor_drift <= v_config.NEIGHBOR_DRIFT_TOLERANCE
        ),
        "general_accuracy_before": general_before,
        "general_accuracy_after": general_after,
        "general_drift": general_drift,
        "general_within_tolerance": (
            general_drift is not None and general_drift <= v_config.GENERAL_DRIFT_TOLERANCE
        ),
        "forget_accuracy_after": forget_after,
        "forget_collapsed": (
            forget_after is not None and forget_after <= v_config.FORGET_ACCURACY_COLLAPSE_THRESHOLD
        ),
    }


def build_key_takeaways(
    manifest_entry: dict,
    impact: dict,
    mia_result: dict,
    reference_result: dict,
    relational_result: dict,
    general_capability_result: dict,
    method_comparison: Optional[dict] = None,
) -> list:
    bullets = []
    bullets.append(
        f"Forget-set accuracy after erasure: {_pct(impact['forget_accuracy_after'])} "
        f"({'collapsed' if impact['forget_collapsed'] else 'did NOT collapse -- investigate'})."
    )
    bullets.append(
        f"Neighbor accuracy: {_pct(impact['neighbor_accuracy_before'])} -> "
        f"{_pct(impact['neighbor_accuracy_after'])} "
        f"({'within tolerance' if impact['neighbor_within_tolerance'] else 'DRIFTED beyond tolerance'})."
    )
    bullets.append(
        f"General-retain accuracy: {_pct(impact['general_accuracy_before'])} -> "
        f"{_pct(impact['general_accuracy_after'])} "
        f"({'within tolerance' if impact['general_within_tolerance'] else 'DRIFTED beyond tolerance'})."
    )

    mia_summary = mia_result["summary"]
    if mia_summary["mean_percentile_rank"] is not None:
        caveat = " (small-sample caveat applies -- see Verification Results)" if mia_summary.get("small_forget_set_caveat") else ""
        bullets.append(
            f"MIA mean percentile rank vs. never-seen text: {mia_summary['mean_percentile_rank']:.2f}{caveat}."
        )

    if reference_result.get("available"):
        acc = reference_result["reference_forget_set_accuracy"]["overall_accuracy"]
        bullets.append(f"Reference (\"never learned it\") model scores {_pct(acc)} on the same forgotten facts.")
    else:
        bullets.append("No retain-only reference model available for this request -- see Verification Results.")

    rel_summary = relational_result["summary"]
    if rel_summary["n_relational_examples_probed"]:
        bullets.append(
            f"Multi-hop probing: forgotten entity dropped out of "
            f"{_pct(rel_summary['forgotten_entity_dropped_out_rate'])} of relational answers; "
            f"retained siblings preserved in {_pct(rel_summary['retained_sibling_preserved_rate'])}."
        )

    gc = general_capability_result["generic_prompts"]["summary"]
    bullets.append(f"General-capability spot-check: {_pct(gc['accuracy'])} on {gc['n']} fixed generic prompts.")

    if method_comparison is not None:
        bullets.append(
            f"{method_comparison['this_method'].upper()} vs {method_comparison['sibling_method'].upper()} "
            f"(revision-{method_comparison['sibling_revision']}): forget accuracy "
            f"{_pct(method_comparison['this_forget_accuracy'])} vs "
            f"{_pct(method_comparison['sibling_forget_accuracy'])}; neighbor accuracy "
            f"{_pct(method_comparison['this_neighbor_accuracy'])} vs "
            f"{_pct(method_comparison['sibling_neighbor_accuracy'])}."
        )

    return bullets


def build_report(
    manifest_entry: dict,
    resolved_summary: dict,
    direct_qa_before: dict,
    direct_qa_after: dict,
    relational_result: dict,
    decoy_results: list,
    mia_result: dict,
    reference_result: dict,
    general_capability_result: dict,
    method_comparison: Optional[dict] = None,
) -> dict:
    impact = build_impact_assessment(direct_qa_before, direct_qa_after)
    takeaways = build_key_takeaways(
        manifest_entry, impact, mia_result, reference_result, relational_result,
        general_capability_result, method_comparison,
    )

    return {
        "revision": manifest_entry["revision"],
        "method": manifest_entry["method"],
        "created_at": manifest_entry["created_at"],
        "what_was_targeted": {
            "erasure_request": manifest_entry["erasure_request"],
            **resolved_summary,
        },
        "what_was_done": {
            "method": manifest_entry["method"],
            "parent_revision": manifest_entry["parent_revision"],
            "training_args": manifest_entry["training_args"],
            "early_stop_step": manifest_entry["early_stop_step"],
        },
        "verification_results": {
            "direct_qa_before": direct_qa_before,
            "direct_qa_after": direct_qa_after,
            "relational_probing": relational_result,
            "decoy_checks": decoy_results,
            "membership_inference": mia_result,
            "reference_model_comparison": reference_result,
            "general_capability": general_capability_result,
            "npo_vs_ga_comparison": method_comparison,
        },
        "impact_assessment": impact,
        "key_takeaways": takeaways,
    }


def _render_md(report: dict) -> str:
    lines = [
        f"# Erasure Report -- revision-{report['revision']} ({report['method']})",
        "",
        f"Created: {report['created_at']}",
        "",
        "## What Was Targeted", "",
        "```json", json.dumps(report["what_was_targeted"], indent=2), "```", "",
        "## What Was Done", "",
        "```json", json.dumps(report["what_was_done"], indent=2), "```", "",
        "## Verification Results", "",
        "```json", json.dumps(report["verification_results"], indent=2), "```", "",
        "## Impact Assessment", "",
        "```json", json.dumps(report["impact_assessment"], indent=2), "```", "",
        "## Key Takeaways", "",
    ]
    lines += [f"- {b}" for b in report["key_takeaways"]]
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, reports_dir: Optional[Path] = None) -> Path:
    reports_dir = reports_dir or v_config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    revision = report["revision"]
    json_path = reports_dir / f"revision-{revision}_verification_report.json"
    md_path = reports_dir / f"revision-{revision}_verification_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_render_md(report))
    return json_path
