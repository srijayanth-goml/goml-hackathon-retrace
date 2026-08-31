"""
Assembles the final unlearning training batches (D_forget / D_retain_general /
D_retain_neighbor) from train.jsonl + selectors.py's resolved request, and applies the
forget-probe split (forget_probe_split.py) so a slice of each targeted fact's own
surface forms is reserved for post-hoc generalization probing rather than trained on
directly.
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

import config as root_config
from data_pipeline.format_chat import read_jsonl
from unlearning import config as ul_config
from unlearning.forget_probe_split import split_forget_probes
from unlearning.request import ErasureRequest
from unlearning.selectors import FactIndex, RecordSets, ResolvedRequest, build_record_sets, load_neighbor_lookup, resolve

Record = dict


def load_train_records() -> List[Record]:
    return list(read_jsonl(root_config.TRAIN_JSONL_PATH))


class UnlearningBatches:
    def __init__(self, resolved: ResolvedRequest, record_sets: RecordSets, forget_train: List[Record], forget_probe: List[Record]):
        self.resolved = resolved
        self.record_sets = record_sets
        self.forget_train = forget_train    # what NPO/GA actually trains against
        self.forget_probe = forget_probe    # held back -- eval only (forget_probe_split.py)
        self.retain_general = record_sets.retain_general
        self.retain_neighbor = record_sets.retain_neighbor

    def summary(self) -> dict:
        return {
            "request": self.resolved.request.to_dict(),
            "entity_type": self.resolved.entity_type,
            "forget_fact_ids": sorted(self.resolved.forget_fact_ids),
            "forget_fact_group_ids": sorted(self.resolved.forget_fact_group_ids),
            "retain_neighbor_entities": sorted(self.resolved.retain_neighbor_entities),
            "n_forget_train": len(self.forget_train),
            "n_forget_probe": len(self.forget_probe),
            "n_retain_neighbor": len(self.retain_neighbor),
            "n_retain_general": len(self.retain_general),
            "n_redacted_relational": len(self.record_sets.redacted_relational),
            "n_redacted_bio": len(self.record_sets.redacted_bio),
            "n_dropped_relational_no_retained_sibling": len(self.record_sets.dropped_relational_no_retained_sibling),
        }


def build_unlearning_batches(
    request: ErasureRequest,
    records: Optional[List[Record]] = None,
    fact_index: Optional[FactIndex] = None,
    neighbor_lookup: Optional[dict] = None,
    seed: Optional[int] = None,
) -> UnlearningBatches:
    records = records if records is not None else load_train_records()
    fact_index = fact_index or FactIndex.load()
    neighbor_lookup = neighbor_lookup or load_neighbor_lookup()

    resolved = resolve(request, neighbor_lookup, fact_index)
    record_sets = build_record_sets(records, resolved, fact_index)
    forget_train, forget_probe = split_forget_probes(record_sets.forget, seed=seed)

    # Guard against a REAL failure mode found while building this module's example
    # requests: Module 1's train/heldout split (data_pipeline/split.py) holds out an
    # entity's ENTIRE fact_group_id -- every paraphrase/qa/bio example -- so an
    # entity that landed in heldout has ZERO of those in train.jsonl, and the only
    # forget-set records found for it are relational examples that merely MENTION it
    # alongside train-split siblings. Unlearning such a request would have nothing
    # genuine to forget (the baseline never learned the target's own facts at all).
    non_relational_forget = [
        r for r in (forget_train + forget_probe) if r["metadata"]["source_type"] != "relational"
    ]
    if not non_relational_forget:
        raise ValueError(
            f"the forget set for {request.to_dict()} has zero paraphrase/qa/bio examples in "
            f"data/processed/train.jsonl -- the target entity/entities are most likely in "
            f"Module 1's HELDOUT split (train/heldout is split by whole fact_group_id, so a "
            f"heldout entity was never trained on at all, only possibly mentioned in a "
            f"relational example about its train-split siblings). There is nothing genuine to "
            f"unlearn here; pick a different entity/attribute, or check which split the target "
            f"landed in via data/processed/build_report.md / neighbor_lookup.json."
        )

    return UnlearningBatches(resolved, record_sets, forget_train, forget_probe)


def forget_sampler(forget_train: List[Record], rng: random.Random) -> Iterator[Record]:
    if not forget_train:
        raise ValueError("forget_train is empty -- nothing to unlearn (did the probe split take everything?)")
    while True:
        yield rng.choice(forget_train)


def neighbor_weighted_sampler(
    retain_general: List[Record], retain_neighbor: List[Record], rng: random.Random
) -> Iterator[Record]:
    """Yields an endless stream of retain records, drawing from `retain_neighbor` at
    a much higher PER-ITEM rate than `retain_general` (Design Doc Section 6:
    "over-samples the confusable neighbors ... far more often than an unrelated ...
    company") -- both pools get the same nominal number of draws per step
    (unlearning/config.py's RETAIN_*_PER_FORGET), but the neighbor pool is usually
    far smaller, so its individual examples repeat much more often per epoch. Falls
    back to whichever pool is non-empty if the other is empty (e.g. an
    attribute-type request scoped to one entity_type has no cross-type
    general-retain need for that type)."""
    if not retain_neighbor and not retain_general:
        raise ValueError("both retain pools are empty -- nothing to retain-train against")
    while True:
        pools = []
        if retain_neighbor:
            pools += [retain_neighbor] * ul_config.RETAIN_NEIGHBOR_PER_FORGET
        if retain_general:
            pools += [retain_general] * ul_config.RETAIN_GENERAL_PER_FORGET
        pool = rng.choice(pools)
        yield rng.choice(pool)
