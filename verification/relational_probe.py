"""
Multi-hop / relational probing (Design Doc Section 7) plus the decoy-mention /
over-forgetting check (review doc's Silvergate Labs case, see
verification/config.py's DECOY_CHECKS). Reuses unlearning.redact.redact_relational_record
directly -- the SAME function unlearning.selectors.build_record_sets calls
internally -- rather than consuming its already-mixed RecordSets.redacted_relational /
dropped_relational_no_retained_sibling output, so every probe here comes with its own
explicit original/redacted PAIR: RecordSets' flat lists don't preserve that pairing
once multiple relational records get mixed together across a whole request.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from finetuning.eval_quick import generate_answer
from unlearning import redact
from unlearning.data import load_train_records
from unlearning.selectors import FactIndex, ResolvedRequest

Record = dict


def _relational_forget_pairs(
    records: List[Record], resolved: ResolvedRequest, fact_index: FactIndex
) -> List[Tuple[Record, Optional[Record]]]:
    """For every relational train.jsonl record that mentions at least one forgotten
    entity: (original_record, redacted_record_or_None). redacted is None when every
    mentioned entity in that example is being forgotten (nothing left to preserve --
    see unlearning/redact.py)."""
    pairs: List[Tuple[Record, Optional[Record]]] = []
    for r in records:
        md = r["metadata"]
        if md.get("source_type") != "relational":
            continue
        mentioned = set(md.get("mentioned_entities") or md.get("fact_group_ids") or [])
        if not (mentioned & resolved.forget_fact_group_ids):
            continue
        redacted = redact.redact_relational_record(r, resolved.forget_fact_group_ids, fact_index.entity_by_group)
        pairs.append((r, redacted))
    return pairs


def probe_relational(
    model,
    tokenizer,
    resolved: ResolvedRequest,
    records: Optional[List[Record]] = None,
    fact_index: Optional[FactIndex] = None,
) -> dict:
    """For each relational example naming a forgotten entity: ask the ORIGINAL
    (un-redacted) question post-erasure and check the forgotten entity no longer
    appears in the answer, then check the retained sibling(s) from the redacted
    counterpart are still present -- Design Doc Section 7's "has the forgotten
    entity actually dropped out of indirect reasoning, not just direct lookup,"
    the exact failure mode the brief calls out prompt-filters for."""
    fact_index = fact_index or FactIndex.load()
    records = records if records is not None else load_train_records()
    pairs = _relational_forget_pairs(records, resolved, fact_index)

    forgotten_names = {fact_index.entity_by_group[g] for g in resolved.forget_fact_group_ids}

    results = []
    for original, redacted in pairs:
        question = original["messages"][0]["content"]
        answer = generate_answer(model, tokenizer, question)
        answer_lower = answer.lower()
        forgotten_still_present = any(name.lower() in answer_lower for name in forgotten_names)

        entry = {
            "question": question,
            "generated_answer": answer,
            "forgotten_entities_in_this_example": sorted(
                {fact_index.entity_by_group[g] for g in (original["metadata"].get("mentioned_entities") or [])}
                & forgotten_names
            ),
            "forgotten_entity_dropped_out": not forgotten_still_present,
        }
        if redacted is not None:
            retained_gids = redacted["metadata"]["mentioned_entities"]
            retained_entity_names = {fact_index.entity_by_group[g] for g in retained_gids}
            retained_all_present = all(n.lower() in answer_lower for n in retained_entity_names)
            entry["retained_siblings"] = sorted(retained_entity_names)
            entry["retained_siblings_still_present"] = retained_all_present
        else:
            entry["retained_siblings"] = []
            entry["retained_siblings_still_present"] = None  # nothing left to check -- fully-forgotten cluster
        results.append(entry)

    n = len(results)
    n_dropped_out = sum(1 for e in results if e["forgotten_entity_dropped_out"])
    scoreable_siblings = [e for e in results if e["retained_siblings_still_present"] is not None]
    n_siblings_ok = sum(1 for e in scoreable_siblings if e["retained_siblings_still_present"])

    summary = {
        "n_relational_examples_probed": n,
        "forgotten_entity_dropped_out_rate": (n_dropped_out / n) if n else None,
        "retained_sibling_preserved_rate": (n_siblings_ok / len(scoreable_siblings)) if scoreable_siblings else None,
    }
    return {"summary": summary, "details": results}


def check_decoys(
    model,
    tokenizer,
    decoy_checks: List[dict],
    records: Optional[List[Record]] = None,
    fact_index: Optional[FactIndex] = None,
) -> List[dict]:
    """Runs verification/config.py's DECOY_CHECKS: for each declared case, finds the
    (unrelated) fact whose value contains the decoy substring, looks up its ACTUAL
    forward-QA record from train.jsonl (never a hand-rolled question template --
    keeps the probe in the model's own trained phrasing style), and confirms it's
    still answered correctly post-erasure -- a different failure direction
    (over-forgetting a name-alike that was never in scope) from probe_relational's
    neighbor-drift check."""
    fact_index = fact_index or FactIndex.load()
    records = records if records is not None else load_train_records()

    forward_qa_by_fact_id = {
        r["metadata"]["fact_ids"][0]: r
        for r in records
        if r["metadata"]["source_type"] == "qa"
        and r["metadata"].get("direction") == "forward"
        and len(r["metadata"].get("fact_ids") or []) == 1
    }

    results = []
    for check in decoy_checks:
        attribute = check["check_attribute"]
        decoy_substring = check["decoy_value_substring"].lower()
        matches = [r for r in fact_index.rows if r.attribute == attribute and decoy_substring in r.value.lower()]
        if not matches:
            results.append({**check, "status": "no_matching_fact_found"})
            continue
        for row in matches:
            qa_record = forward_qa_by_fact_id.get(row.fact_id)
            if qa_record is None:
                results.append({**check, "probed_entity": row.entity, "status": "no_forward_qa_record_in_train_jsonl"})
                continue
            question = qa_record["messages"][0]["content"]
            answer = generate_answer(model, tokenizer, question)
            is_correct = row.value.strip().lower() in answer.strip().lower()
            results.append({
                **check,
                "probed_entity": row.entity,
                "expected_value": row.value,
                "question": question,
                "generated_answer": answer,
                "correct": is_correct,
            })
    return results
