"""
Comparison against Module 2's retain-only reference model (Design Doc Section 6:
"behaves like it never learned this, not just no longer says the target answer") --
the strongest available verification signal where it exists. Only ever available
today for a request whose forget_fact_group_ids intersect
finetuning/checkpoints/manifest.json["reference_models"] (currently just NeuroSync
Diagnostics / G001) -- every other request gets an explicit "unavailable" result,
never a silently omitted section. Locked recommendation (plan.md's Module 4 Open
Decisions): re-run the comparison fresh rather than trusting the reference model's
own stored eval_summary, same arm's-length principle as everywhere else in this
module.
"""
from __future__ import annotations

from typing import List, Optional

from finetuning import manifest as ft_manifest
from unlearning.model_io import load_single_adapter
from unlearning.selectors import FactIndex, ResolvedRequest
from verification import config as v_config
from verification.direct_qa import accuracy_on

Record = dict


def find_matching_reference_model(resolved: ResolvedRequest, manifest: Optional[dict] = None) -> Optional[dict]:
    """A request's forget_fact_group_ids may span many entities (an attribute-type
    request forgets all 53 companies' CEO facts, say); this only checks whether ANY
    of them has reference-model coverage. compare_against_reference below then
    scopes the actual comparison down to just that one entity's forget records --
    the reference model is retain-only for ONE entity, not for the rest of a larger
    forget set."""
    manifest = manifest if manifest is not None else ft_manifest.read_manifest()
    for entry in manifest.get("reference_models", []):
        if entry["fact_group_id"] in resolved.forget_fact_group_ids:
            return entry
    return None


def compare_against_reference(
    resolved: ResolvedRequest,
    forget_records: List[Record],
    fact_index: Optional[FactIndex] = None,
    manifest: Optional[dict] = None,
) -> dict:
    fact_index = fact_index or FactIndex.load()
    entry = find_matching_reference_model(resolved, manifest)
    if entry is None:
        return {
            "available": False,
            "reason": (
                "no retain-only reference model exists for any of this request's forgotten "
                f"entities ({sorted(resolved.forget_fact_group_ids)}) -- Module 2 built exactly "
                "one reference model (NeuroSync Diagnostics / G001); see plan.md's Module 2 Open "
                "Decisions for why. This request falls back to self-comparison + neighbor/general "
                "accuracy only, one signal weaker than the NeuroSync demo gets."
            ),
        }

    reference_fact_group_id = entry["fact_group_id"]
    scoped_records = [
        r for r in forget_records
        if reference_fact_group_id in (r["metadata"].get("fact_group_ids") or [])
    ]
    if not scoped_records:
        return {
            "available": False,
            "reason": (
                f"a reference model exists for {entry['entity']} ({reference_fact_group_id}) and its "
                f"fact_group_id is among this request's forgotten groups, but none of the forget "
                f"records passed in are actually about that entity -- check the caller."
            ),
        }

    model, tokenizer = load_single_adapter(entry["adapter_path"], v_config.MODEL_NAME, bf16=True)
    summary, details = accuracy_on(model, tokenizer, scoped_records, fact_index)
    return {
        "available": True,
        "reference_entity": entry["entity"],
        "reference_fact_group_id": reference_fact_group_id,
        "reference_adapter_path": entry["adapter_path"],
        "scoped_to_n_forget_records": len(scoped_records),
        "reference_forget_set_accuracy": summary,
        "reference_forget_set_details": details,
        "note": (
            f"scored ONLY the subset of this request's forget set belonging to {entry['entity']} "
            "-- the reference model is retain-only for that one entity, not for the rest of this "
            "request's forget set (relevant for attribute-type requests, which forget many "
            "entities' facts at once but only ever have reference-model coverage for one of them)."
        ),
    }
