"""
Module 4 (Verification & Erasure Report) entrypoint:

    python -m verification.run_verification --revision 1

Loads revision N's adapter (post-erasure) AND its parent_revision's adapter
(pre-erasure) fresh -- NOT the manifest's own self-reported accuracy_before/after --
runs every Design Doc Section 7 signal, and writes the Erasure Report. See
plan.md's Module 4 "Why this module can't just reuse Module 3's own accuracy
tracking" for why this recomputes rather than relays. If a sibling revision exists
for the SAME erasure_request with a DIFFERENT method (npo vs ga), also runs the
side-by-side comparison Design Doc Section 6 wants.

Needs torch/transformers/peft installed (repo-root requirements.txt) -- NOT
necessarily a GPU, same posture as unlearning/train.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from finetuning import ft_config
from unlearning import manifest as ul_manifest
from unlearning.data import build_unlearning_batches, load_train_records
from unlearning.model_io import load_single_adapter
from unlearning.request import ErasureRequest
from unlearning.selectors import FactIndex, build_record_sets, load_neighbor_lookup, resolve
from verification import config as v_config
from verification import direct_qa, general_capability, mia, reference_comparison, relational_probe
from verification import report as report_mod


def _require_heavy_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"verification/run_verification.py needs {missing} installed -- see the "
            f"repo-root requirements.txt and run `pip install -r requirements.txt`. This "
            f"does NOT require a GPU (same posture as unlearning/train.py)."
        )


def _find_sibling_revision(manifest_entry: dict, entries_by_revision: dict) -> Optional[dict]:
    """A sibling is a DIFFERENT revision, same parent_revision, same erasure_request,
    DIFFERENT method -- i.e. the npo/ga pair unlearning/train.py produces when both
    are run for the same request (plan.md's Module 3 step 6)."""
    for rev, entry in entries_by_revision.items():
        if rev == manifest_entry["revision"]:
            continue
        if (
            entry.get("parent_revision") == manifest_entry.get("parent_revision")
            and entry.get("erasure_request") == manifest_entry.get("erasure_request")
            and entry.get("method") != manifest_entry.get("method")
        ):
            return entry
    return None


def run(revision: int) -> Path:
    _require_heavy_deps()

    manifest = ul_manifest.read_manifest()
    entries_by_revision = {e["revision"]: e for e in manifest["revisions"]}
    if revision not in entries_by_revision:
        raise KeyError(f"revision {revision} not found in finetuning/checkpoints/manifest.json")
    manifest_entry = entries_by_revision[revision]
    parent_revision = manifest_entry.get("parent_revision")
    if parent_revision is None:
        raise ValueError(
            f"revision {revision} has no parent_revision -- it IS revision-0, the baseline. "
            f"Verification compares an erasure run against its parent; pick a revision "
            f"produced by `python -m unlearning.train` instead."
        )
    if parent_revision not in entries_by_revision:
        raise KeyError(f"parent_revision {parent_revision} not found in the manifest")
    parent_entry = entries_by_revision[parent_revision]

    request = ErasureRequest.from_dict(manifest_entry["erasure_request"])
    fact_index = FactIndex.load()
    neighbor_lookup = load_neighbor_lookup()
    train_records = load_train_records()
    resolved = resolve(request, neighbor_lookup, fact_index)
    record_sets = build_record_sets(train_records, resolved, fact_index)
    batches = build_unlearning_batches(request, records=train_records, fact_index=fact_index, neighbor_lookup=neighbor_lookup)

    pre_model, pre_tokenizer = load_single_adapter(parent_entry["adapter_path"], v_config.MODEL_NAME, bf16=ft_config.BF16)
    post_model, post_tokenizer = load_single_adapter(manifest_entry["adapter_path"], v_config.MODEL_NAME, bf16=ft_config.BF16)

    common_kwargs = dict(
        records=train_records, fact_index=fact_index, neighbor_lookup=neighbor_lookup,
        resolved=resolved, record_sets=record_sets, batches=batches,
    )
    print("[verification] scoring pre-erasure (parent) model...")
    direct_qa_before = direct_qa.three_way_accuracy(pre_model, pre_tokenizer, request, **common_kwargs)
    print("[verification] scoring post-erasure model...")
    direct_qa_after = direct_qa.three_way_accuracy(post_model, post_tokenizer, request, **common_kwargs)

    print("[verification] multi-hop / relational probing...")
    relational_result = relational_probe.probe_relational(
        post_model, post_tokenizer, resolved, records=train_records, fact_index=fact_index
    )
    decoy_results = relational_probe.check_decoys(
        post_model, post_tokenizer, v_config.DECOY_CHECKS, records=train_records, fact_index=fact_index
    )

    print("[verification] loss-based membership inference...")
    mia_result = mia.run_mia(post_model, post_tokenizer, record_sets.forget, max_length=ft_config.MAX_SEQ_LENGTH)

    print("[verification] reference-model comparison...")
    reference_result = reference_comparison.compare_against_reference(resolved, record_sets.forget, fact_index=fact_index)

    print("[verification] general-capability spot-check...")
    general_capability_result = general_capability.run_general_capability(
        post_model, post_tokenizer, records=train_records, fact_index=fact_index
    )

    method_comparison = None
    sibling_entry = _find_sibling_revision(manifest_entry, entries_by_revision)
    if sibling_entry is not None:
        print(f"[verification] found sibling revision-{sibling_entry['revision']} ({sibling_entry['method']}) -- comparing...")
        sibling_model, sibling_tokenizer = load_single_adapter(
            sibling_entry["adapter_path"], v_config.MODEL_NAME, bf16=ft_config.BF16
        )
        sibling_direct_qa_after = direct_qa.three_way_accuracy(sibling_model, sibling_tokenizer, request, **common_kwargs)
        method_comparison = {
            "this_method": manifest_entry["method"],
            "this_revision": manifest_entry["revision"],
            "this_forget_accuracy": direct_qa_after["forget"]["overall_accuracy"],
            "this_neighbor_accuracy": direct_qa_after["retain_neighbor"]["overall_accuracy"],
            "sibling_method": sibling_entry["method"],
            "sibling_revision": sibling_entry["revision"],
            "sibling_forget_accuracy": sibling_direct_qa_after["forget"]["overall_accuracy"],
            "sibling_neighbor_accuracy": sibling_direct_qa_after["retain_neighbor"]["overall_accuracy"],
        }

    resolved_summary = {
        "request_type": request.request_type,
        "entity_type": resolved.entity_type,
        "forget_fact_ids": sorted(resolved.forget_fact_ids),
        "forget_fact_group_ids": sorted(resolved.forget_fact_group_ids),
        "retain_neighbor_entities": sorted(resolved.retain_neighbor_entities),
    }

    report = report_mod.build_report(
        manifest_entry=manifest_entry,
        resolved_summary=resolved_summary,
        direct_qa_before=direct_qa_before,
        direct_qa_after=direct_qa_after,
        relational_result=relational_result,
        decoy_results=decoy_results,
        mia_result=mia_result,
        reference_result=reference_result,
        general_capability_result=general_capability_result,
        method_comparison=method_comparison,
    )
    path = report_mod.write_report(report)
    print(f"[verification] report written to {path}")
    return path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, required=True, help="Manifest revision to verify (must have a parent_revision, i.e. not revision-0)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.revision)


if __name__ == "__main__":
    main()
