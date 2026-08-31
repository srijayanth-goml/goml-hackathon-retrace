"""
General-capability spot-check (Design Doc Section 7): (a) a small fixed set of
mechanically-gradable, KB-independent prompts (verification/config.py's
GENERAL_CAPABILITY_PROMPTS) to catch wholesale model degradation a narrow retain-set
check might miss; (b) the previous_company control group (review doc: 12 real-world
brands -- Bosch, Philips, Siemens, etc. -- never colliding with anything synthetic in
this dataset) scored via the SAME forward-QA scoring direct_qa.py uses everywhere
else -- a KB-grounded check in the model's own trained phrasing style, which can't
conflate "the fine-tune narrowed general ability" with "erasure did" the way (a)
alone could.
"""
from __future__ import annotations

from typing import List, Optional

from finetuning.eval_quick import generate_answer
from unlearning.data import load_train_records
from unlearning.selectors import FactIndex
from verification import config as v_config
from verification.direct_qa import accuracy_on

Record = dict


def run_fixed_prompts(model, tokenizer, prompts: Optional[List[dict]] = None) -> dict:
    prompts = prompts if prompts is not None else v_config.GENERAL_CAPABILITY_PROMPTS
    results = []
    for item in prompts:
        answer = generate_answer(model, tokenizer, item["prompt"])
        is_correct = item["expected_substring"].strip().lower() in answer.strip().lower()
        results.append({**item, "generated_answer": answer, "correct": is_correct})
    n_correct = sum(1 for r in results if r["correct"])
    return {
        "summary": {
            "n": len(results),
            "n_correct": n_correct,
            "accuracy": (n_correct / len(results)) if results else None,
        },
        "details": results,
    }


def previous_company_control_group(
    model, tokenizer, records: Optional[List[Record]] = None, fact_index: Optional[FactIndex] = None
) -> dict:
    fact_index = fact_index or FactIndex.load()
    records = records if records is not None else load_train_records()
    previous_company_records = [
        r for r in records
        if r["metadata"]["source_type"] in ("qa", "paraphrase") and r["metadata"]["attribute"] == "previous_company"
    ]
    summary, details = accuracy_on(model, tokenizer, previous_company_records, fact_index)
    return {"summary": summary, "details": details}


def run_general_capability(
    model, tokenizer, records: Optional[List[Record]] = None, fact_index: Optional[FactIndex] = None
) -> dict:
    return {
        "generic_prompts": run_fixed_prompts(model, tokenizer),
        "previous_company_control_group": previous_company_control_group(model, tokenizer, records, fact_index),
    }
