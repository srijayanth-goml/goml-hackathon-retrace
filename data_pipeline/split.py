"""
Train/held-out split, by `fact_group_id` -- never by row (see ../CLAUDE.md: a
row-level split would let the model see 4 of an entity's 5 facts in training and
"predict" the held-out fifth, inflating apparent generalization). Held-out is
reserved purely as an eval-time probe set for Module 4 and must never be trained on.

Also decides where multi-entity relational examples land when their mentioned
fact_group_ids straddle both splits: held-out only if ALL mentioned groups are
held-out, otherwise train. (Decision from plan.md's open-decisions list.)
"""
from __future__ import annotations

import random
from typing import Dict, List

from common.schema import ChatExample


def assign_group_splits(
    fact_group_ids_by_type: Dict[str, List[str]],
    heldout_fraction: float,
    seed: int,
) -> Dict[str, str]:
    """Stratified by entity_type so held-out keeps roughly the company/person mix."""
    rng = random.Random(seed)
    split_of: Dict[str, str] = {}
    for entity_type, gids in fact_group_ids_by_type.items():
        gids = sorted(gids)  # deterministic order before shuffling
        rng.shuffle(gids)
        n_heldout = round(len(gids) * heldout_fraction)
        heldout_gids = set(gids[:n_heldout])
        for gid in gids:
            split_of[gid] = "heldout" if gid in heldout_gids else "train"
    return split_of


def assign_example_splits(examples: List[ChatExample], split_of_group: Dict[str, str]) -> None:
    """Mutates each example's metadata.split in place."""
    for ex in examples:
        gids = ex.metadata.fact_group_ids
        if len(gids) == 1:
            ex.metadata.split = split_of_group[gids[0]]
        else:
            # Relational example spanning multiple entities.
            splits = {split_of_group[g] for g in gids}
            ex.metadata.split = "heldout" if splits == {"heldout"} else "train"
