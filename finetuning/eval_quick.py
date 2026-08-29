"""
Module 2's post-training quick sanity check (plan.md step 6) -- NOT Module 4's full
verification suite. This exists purely as a fast, cheap gate that answers "did this
training run actually work" before spending more Colab time or handing off to
Module 3: forward-direction QA accuracy against data/processed/heldout.jsonl
(read-only, never trained on), overall and broken out by attribute.

Expect the baseline (revision-0) to score high across the board. Expect the
retain-only reference model to score at-or-near-chance specifically on the flagship
demo entity's own held-out QA (it never saw those facts) while matching the baseline
everywhere else -- that gap is itself informal evidence the retain-only training set
was built correctly.

Only `run_quick_eval`'s generation step needs a real transformers model+tokenizer at
call time; everything else here (loading heldout QA, looking up expected values)
needs only the repo-root requirements.txt.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import config as root_config
from data_pipeline.format_chat import read_jsonl
from data_pipeline.load import load_fact_rows


def _fact_value_map() -> Dict[str, str]:
    """fact_id -> ground-truth value, straight from the source CSV (not from the
    training text) -- so eval doesn't accidentally grade against paraphrased wording."""
    rows = load_fact_rows(root_config.RAW_CSV_PATH)
    return {r.fact_id: r.value for r in rows}


def load_forward_qa_heldout() -> List[dict]:
    records = list(read_jsonl(root_config.HELDOUT_JSONL_PATH))
    return [
        r for r in records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "forward"
    ]


def generate_answer(model, tokenizer, user_message: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_quick_eval(model, tokenizer) -> Tuple[dict, List[dict]]:
    """Returns (summary, per_example_details). summary has overall_accuracy,
    accuracy_by_attribute, and n; per_example_details records question/expected/got/
    correct for every held-out forward-QA example, for manual spot-checking."""
    value_map = _fact_value_map()
    records = load_forward_qa_heldout()

    correct_by_attr: Counter = Counter()
    total_by_attr: Counter = Counter()
    details: List[dict] = []

    for r in records:
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
            "fact_id": fact_id,
            "entity": r["metadata"]["entity"],
            "attribute": attribute,
            "question": question,
            "expected_value": expected_value,
            "generated_answer": answer,
            "correct": is_correct,
        })

    accuracy_by_attribute = {
        attr: correct_by_attr[attr] / total_by_attr[attr] for attr in sorted(total_by_attr)
    }
    n = sum(total_by_attr.values())
    overall_accuracy = (sum(correct_by_attr.values()) / n) if n else 0.0

    summary = {
        "n": n,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_attribute": accuracy_by_attribute,
    }
    return summary, details
