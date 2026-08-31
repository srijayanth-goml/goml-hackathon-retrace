"""
Direct-QA accuracy harness for tracking forget/neighbor/general-retain accuracy
during and after an unlearning run (Design Doc Section 6 step 4 / plan.md's early-
stop-on-neighbor-drift rule). Queries TRAIN-split facts directly -- NOT
data/processed/heldout.jsonl -- since every entity this module ever operates on is,
by definition, a train-split entity (an entity the model never learned can't be
meaningfully unlearned); see plan.md's "Result of the actual run" note under Module 2
for why finetuning/eval_quick.py's heldout-based check can't supply this signal.
Reuses finetuning/eval_quick.py's generate_answer rather than reimplementing
generation.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import config as root_config
from data_pipeline.load import load_fact_rows
from finetuning.eval_quick import generate_answer

Record = dict


def _fact_value_map() -> Dict[str, str]:
    rows = load_fact_rows(root_config.RAW_CSV_PATH)
    return {r.fact_id: r.value for r in rows}


def _forward_qa_only(records: List[Record]) -> List[Record]:
    return [
        r for r in records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "forward"
    ]


def accuracy_on(model, tokenizer, records: List[Record]) -> Tuple[dict, List[dict]]:
    """Forward-QA exact-match accuracy over `records` (any list of train.jsonl-shaped
    records -- a forget set, a neighbor set, a forget-probe set, or a general-retain
    sample) -- same exact-match logic as finetuning/eval_quick.py's run_quick_eval,
    generalized to take an arbitrary record list instead of always reading
    heldout.jsonl. Returns (summary, per_example_details); overall_accuracy is None
    (not 0.0) when `records` has zero forward-QA examples, so callers can tell "no
    signal" apart from "measured and it's zero"."""
    value_map = _fact_value_map()
    qa_records = _forward_qa_only(records)

    correct_by_attr: Counter = Counter()
    total_by_attr: Counter = Counter()
    details: List[dict] = []

    for r in qa_records:
        fact_id = r["metadata"]["fact_ids"][0]
        attribute = r["metadata"]["attribute"]
        expected_value = value_map[fact_id]
        question = r["messages"][0]["content"]

        answer = generate_answer(model, tokenizer, question)
        is_correct = expected_value.strip().lower() in answer.strip().lower()

        total_by_attr[attribute] += 1
        if is_correct:
            correct_by_attr[attribute] += 1
        details.append({
            "fact_id": fact_id, "entity": r["metadata"]["entity"], "attribute": attribute,
            "question": question, "expected_value": expected_value,
            "generated_answer": answer, "correct": is_correct,
        })

    n = sum(total_by_attr.values())
    summary = {
        "n": n,
        "overall_accuracy": (sum(correct_by_attr.values()) / n) if n else None,
        "accuracy_by_attribute": {a: correct_by_attr[a] / total_by_attr[a] for a in sorted(total_by_attr)},
    }
    return summary, details


def track_all(model, tokenizer, forget_records, neighbor_records, general_records, probe_records) -> dict:
    """Runs accuracy_on for all four pools in one call -- the exact snapshot
    unlearning/train.py's early-stop-on-neighbor-drift rule compares against its
    pre-unlearning baseline every EVAL_EVERY_N_STEPS. `general_records` should
    normally be a fixed-size SAMPLE of the full retain-general pool (see
    unlearning/train.py's _sample_general), not the whole ~2,200-example pool, so
    this stays fast enough to run every few training steps."""
    forget_summary, _ = accuracy_on(model, tokenizer, forget_records)
    neighbor_summary, _ = accuracy_on(model, tokenizer, neighbor_records)
    general_summary, _ = accuracy_on(model, tokenizer, general_records)
    probe_summary, _ = accuracy_on(model, tokenizer, probe_records)
    return {
        "forget": forget_summary,
        "neighbor": neighbor_summary,
        "general": general_summary,
        "forget_probe": probe_summary,
    }
