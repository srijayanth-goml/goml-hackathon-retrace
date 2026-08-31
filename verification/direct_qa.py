"""
Direct-QA accuracy scoring, broadened beyond unlearning/eval_during_unlearning.py's
forward-QA-only scoring (see plan.md's Module 4 "Why this module can't just reuse
Module 3's own accuracy tracking"): also scores paraphrase-source_type records and
reverse-direction QA, and splits the result three ways (forget / retain-neighbor /
retain-general="unrelated") by reusing unlearning.selectors.resolve + build_record_sets
directly, so this module's notion of "neighbor" and "unrelated" can never drift from
what unlearning/train.py actually trained against.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from finetuning.eval_quick import generate_answer
from unlearning.data import build_unlearning_batches, load_train_records
from unlearning.request import ErasureRequest
from unlearning.selectors import (
    FactIndex,
    RecordSets,
    ResolvedRequest,
    build_record_sets,
    load_neighbor_lookup,
    resolve,
)

Record = dict


def _fact_value_map(fact_index: Optional[FactIndex] = None) -> Dict[str, str]:
    fact_index = fact_index or FactIndex.load()
    return {r.fact_id: r.value for r in fact_index.rows}


def _scoreable_records(records: List[Record]) -> List[Record]:
    """paraphrase records + BOTH directions of qa -- unlike
    unlearning.eval_during_unlearning.accuracy_on, which only keeps forward qa."""
    return [
        r for r in records
        if r["metadata"]["source_type"] == "paraphrase"
        or (r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] in ("forward", "reverse"))
    ]


def _expected_answer(record: Record, value_map: Dict[str, str]) -> Optional[str]:
    """Forward QA / paraphrase: expected = the fact's VALUE. Reverse QA: expected =
    the ENTITY NAME -- a different lookup, worth its own branch so a reverse record
    is never silently scored against the wrong string (plan.md's Module 4 step 2)."""
    md = record["metadata"]
    if md["source_type"] == "qa" and md.get("direction") == "reverse":
        return md.get("entity")
    fact_ids = md.get("fact_ids") or []
    if len(fact_ids) != 1:
        return None  # be defensive, not silent -- every paraphrase/forward-qa record here carries exactly one
    return value_map.get(fact_ids[0])


def accuracy_on(model, tokenizer, records: List[Record], fact_index: Optional[FactIndex] = None) -> Tuple[dict, List[dict]]:
    """Broadened sibling of unlearning.eval_during_unlearning.accuracy_on: scores
    paraphrase + forward-QA + reverse-QA records (not forward-QA only), using
    substring-containment against the correct expected string per record type.
    Returns (summary, per_example_details); overall_accuracy is None (not 0.0) when
    `records` has zero scoreable examples, so callers can tell "no signal" apart
    from "measured and it's zero"."""
    value_map = _fact_value_map(fact_index)
    scoreable = _scoreable_records(records)

    correct_by_attr: Counter = Counter()
    total_by_attr: Counter = Counter()
    details: List[dict] = []

    for r in scoreable:
        md = r["metadata"]
        attribute = md["attribute"]
        expected = _expected_answer(r, value_map)
        if expected is None:
            continue
        question = r["messages"][0]["content"]
        answer = generate_answer(model, tokenizer, question)
        is_correct = expected.strip().lower() in answer.strip().lower()

        total_by_attr[attribute] += 1
        if is_correct:
            correct_by_attr[attribute] += 1
        details.append({
            "source_type": md["source_type"],
            "direction": md.get("direction"),
            "entity": md.get("entity"),
            "attribute": attribute,
            "question": question,
            "expected": expected,
            "generated_answer": answer,
            "correct": is_correct,
        })

    n = sum(total_by_attr.values())
    summary = {
        "n": n,
        "overall_accuracy": (sum(correct_by_attr.values()) / n) if n else None,
        "accuracy_by_attribute": {a: correct_by_attr[a] / total_by_attr[a] for a in sorted(total_by_attr)},
    }
    return summary, details


def three_way_accuracy(
    model,
    tokenizer,
    request: ErasureRequest,
    records: Optional[List[Record]] = None,
    fact_index: Optional[FactIndex] = None,
    neighbor_lookup: Optional[dict] = None,
    resolved: Optional[ResolvedRequest] = None,
    record_sets: Optional[RecordSets] = None,
    batches=None,
) -> dict:
    """Splits train.jsonl into forget / retain_neighbor / retain_general="unrelated"
    via the SAME unlearning.selectors resolution unlearning/train.py used, and scores
    each pool plus the forget-probe slice (unlearning.data.build_unlearning_batches'
    forget_probe -- the exact phrasings NPO/GA never trained against, not a
    re-derived split). Accepts pre-built resolved/record_sets/batches so a caller
    already holding them (verification/run_verification.py, scoring both a pre- and
    post-erasure model) doesn't recompute the same request resolution twice."""
    fact_index = fact_index or FactIndex.load()
    neighbor_lookup = neighbor_lookup or load_neighbor_lookup()
    records = records if records is not None else load_train_records()

    resolved = resolved or resolve(request, neighbor_lookup, fact_index)
    record_sets = record_sets or build_record_sets(records, resolved, fact_index)
    batches = batches or build_unlearning_batches(
        request, records=records, fact_index=fact_index, neighbor_lookup=neighbor_lookup
    )

    forget_summary, forget_details = accuracy_on(model, tokenizer, record_sets.forget, fact_index)
    neighbor_summary, _ = accuracy_on(model, tokenizer, record_sets.retain_neighbor, fact_index)
    general_summary, _ = accuracy_on(model, tokenizer, record_sets.retain_general, fact_index)
    probe_summary, probe_details = accuracy_on(model, tokenizer, batches.forget_probe, fact_index)

    return {
        "forget": forget_summary,
        "retain_neighbor": neighbor_summary,
        "retain_general_unrelated": general_summary,
        "forget_probe": probe_summary,
        "forget_details": forget_details,
        "forget_probe_details": probe_details,
    }
