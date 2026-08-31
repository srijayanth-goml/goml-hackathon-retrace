"""
Holds back a small slice of each targeted fact's own surface forms (paraphrase / qa)
from the unlearning training batch entirely, so post-training evaluation can check
whether forgetting generalized to phrasings NPO/GA never directly trained against.

This closes a gap data_pipeline's entity-level heldout.jsonl cannot fill: Design Doc
Section 7 wants "paraphrase and reverse-direction robustness on the forget set, using
held-out probes never trained on in any phrasing", but every entity this module
operates on is, by definition, a TRAIN-split entity (an entity the model never
learned in the first place can't be meaningfully unlearned) -- see plan.md's Module 3
step 4 for the full reasoning. This is this module's OWN train/probe split, distinct
from Module 1's train/heldout split and Module 2's sft-train/sft-val split -- named
distinctly ("forget-train" / "forget-probe") so the three don't get confused.

bio and relational records are never held back as probes here (there's exactly one
bio paragraph per entity, and relational examples are already handled specially by
selectors.py's redaction logic) -- only paraphrase/qa surface forms, which is where a
fact genuinely has multiple interchangeable phrasings to spare.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from unlearning import config as ul_config

Record = dict


def split_forget_probes(forget_records: List[Record], seed: int = None) -> Tuple[List[Record], List[Record]]:
    """Returns (forget_train, forget_probe). Groups paraphrase/qa records by fact_id
    and holds back max(FORGET_PROBE_MIN_PER_FACT, round(FORGET_PROBE_FRACTION * n)) of
    each fact's surface forms -- but never all of them, so every fact keeps at least
    one example to actually train the forgetting on. bio/relational records (and any
    record whose fact_ids don't resolve to exactly one fact) always stay in
    forget_train untouched. Deterministic under `seed` (defaults to
    unlearning/config.py's FORGET_PROBE_SEED)."""
    seed = ul_config.FORGET_PROBE_SEED if seed is None else seed
    rng = random.Random(seed)

    by_fact: Dict[str, List[Record]] = defaultdict(list)
    untouched: List[Record] = []
    for r in forget_records:
        md = r["metadata"]
        if md["source_type"] not in ("paraphrase", "qa"):
            untouched.append(r)
            continue
        fact_ids = md.get("fact_ids") or []
        if len(fact_ids) != 1:
            untouched.append(r)
            continue
        by_fact[fact_ids[0]].append(r)

    forget_train: List[Record] = list(untouched)
    forget_probe: List[Record] = []
    for fact_id in sorted(by_fact):
        recs = list(by_fact[fact_id])
        rng.shuffle(recs)
        n_probe = max(ul_config.FORGET_PROBE_MIN_PER_FACT, round(ul_config.FORGET_PROBE_FRACTION * len(recs)))
        n_probe = min(n_probe, len(recs) - 1) if len(recs) > 1 else 0
        forget_probe.extend(recs[:n_probe])
        forget_train.extend(recs[n_probe:])

    return forget_train, forget_probe
