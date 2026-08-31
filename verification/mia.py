"""
Loss-based membership inference (Design Doc Section 7): scores forget-set examples'
per-example log-likelihood under the model being verified, and ranks each against a
null distribution built from data/processed/heldout.jsonl's forward-QA records --
genuinely never-trained-on text in the identical chat-formatted shape (CLAUDE.md's
guarantee about heldout.jsonl is exactly what makes this a fair null). Reuses
unlearning.npo.compute_batch_logps and unlearning.model_io.pad_collate rather than
reimplementing the forward-pass/masking logic. Locked recommendation (plan.md's
Module 4 Open Decisions): percentile-rank, not a fitted likelihood-ratio test --
simple, and honest about what it can and can't claim on tiny forget sets.
"""
from __future__ import annotations

import random
from bisect import bisect_left
from typing import List, Optional

import config as root_config
from data_pipeline.format_chat import read_jsonl
from unlearning.model_io import pad_collate
from unlearning.npo import compute_batch_logps
from verification import config as v_config

Record = dict


def _heldout_forward_qa_sample(n: int, seed: int) -> List[Record]:
    records = [
        r for r in read_jsonl(root_config.HELDOUT_JSONL_PATH)
        if r["metadata"]["source_type"] == "qa" and r["metadata"].get("direction") == "forward"
    ]
    rng = random.Random(seed)
    if len(records) <= n:
        return records
    return rng.sample(records, n)


def _logps_for(model, tokenizer, records: List[Record], max_length: int) -> List[float]:
    if not records:
        return []
    import torch

    batch = pad_collate(records, tokenizer, max_length, tokenizer.pad_token_id)
    with torch.no_grad():
        logps = compute_batch_logps(model, batch)
    return [float(x) for x in logps]


def percentile_rank(value: float, null_sorted: List[float]) -> float:
    """Fraction of the null distribution AT OR BELOW `value` -- 0.0 means the
    example looks LESS memorized than anything genuinely unseen (strong evidence of
    forgetting), 1.0 means it looks MORE memorized than everything unseen (evidence
    it's still memorized). `null_sorted` must already be sorted ascending."""
    if not null_sorted:
        raise ValueError("null_sorted is empty -- nothing to rank against")
    idx = bisect_left(null_sorted, value)
    return idx / len(null_sorted)


def run_mia(
    model,
    tokenizer,
    forget_records: List[Record],
    max_length: int,
    null_sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    null_sample_size = v_config.MIA_NULL_SAMPLE_SIZE if null_sample_size is None else null_sample_size
    seed = v_config.MIA_SEED if seed is None else seed

    null_records = _heldout_forward_qa_sample(null_sample_size, seed)
    null_logps = sorted(_logps_for(model, tokenizer, null_records, max_length))

    forget_logps = _logps_for(model, tokenizer, forget_records, max_length)
    ranks = [percentile_rank(lp, null_logps) for lp in forget_logps] if null_logps else []

    small_sample = len(forget_records) < v_config.MIA_MIN_FORGET_SET_FOR_CONFIDENCE
    sorted_ranks = sorted(ranks)
    summary = {
        "n_forget_examples_scored": len(forget_logps),
        "n_null_examples": len(null_logps),
        "mean_percentile_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_percentile_rank": sorted_ranks[len(sorted_ranks) // 2] if sorted_ranks else None,
        "small_forget_set_caveat": (
            f"forget set has only {len(forget_records)} example(s) (< "
            f"{v_config.MIA_MIN_FORGET_SET_FOR_CONFIDENCE}) -- this is a single point (or a "
            f"handful) ranked against a distribution, not a distribution of its own; report "
            f"the rank as an anecdote, not a statistically confident claim (per the review "
            f"doc's own caution about the 1-fact attribute-cell case)."
            if small_sample else None
        ),
    }
    return {"summary": summary, "per_example_ranks": ranks}
