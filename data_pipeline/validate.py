"""
Post-build sanity checks. build_dataset.py exits non-zero if any hard check here
fails, so a broken build is caught at build time, not discovered three modules
downstream.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from common.schema import ChatExample, FactRow
from data_pipeline.neighbors import NeighborLookup

# Regression check from the review doc: these five companies share a name root
# ("Crescent") but span five DIFFERENT industries, so field-based retain-neighbor
# logic must NOT treat them as mutual neighbors. If this ever fails, someone made
# neighbor computation name-based again.
DECORRELATED_NAME_CLUSTER = [
    "Crescent Therapeutics",
    "Crescent Logistics",
    "Crescent Materials",
    "Crescent Analytics",
    "Crescent Energy",
]


def _check_splits_assigned(split_of_group: Dict[str, str]) -> List[str]:
    failures = []
    values = set(split_of_group.values())
    if "train" not in values:
        failures.append("No fact_group_id assigned to the train split.")
    if "heldout" not in values:
        failures.append("No fact_group_id assigned to the heldout split.")
    return failures


def _check_single_entity_split_consistency(
    examples: List[ChatExample], split_of_group: Dict[str, str]
) -> List[str]:
    """Direct leakage guard: a single-entity example's split must match its entity's
    assigned split. (Relational examples spanning multiple entities are exempt --
    see split.py's documented policy for those.)"""
    failures = []
    for ex in examples:
        gids = ex.metadata.fact_group_ids
        if len(gids) == 1 and ex.metadata.split != split_of_group[gids[0]]:
            failures.append(
                f"Example (source_type={ex.metadata.source_type}, fact_ids={ex.metadata.fact_ids}) "
                f"has split={ex.metadata.split!r} but its entity {gids[0]} is assigned "
                f"{split_of_group[gids[0]]!r}."
            )
    return failures


def _check_source_type_coverage(examples: List[ChatExample]) -> List[str]:
    failures = []
    counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.metadata.source_type] += 1
    for source_type in ("paraphrase", "qa", "bio", "relational"):
        if counts[source_type] == 0:
            failures.append(f"source_type={source_type} produced zero examples.")
    return failures


def _check_fact_id_coverage(fact_rows: List[FactRow], examples: List[ChatExample]) -> List[str]:
    """Every fact_id must have at least one paraphrase and one forward-QA example
    somewhere in the built corpus (train OR heldout)."""
    failures = []
    have_paraphrase = set()
    have_forward_qa = set()
    for ex in examples:
        for fid in ex.metadata.fact_ids:
            if ex.metadata.source_type == "paraphrase":
                have_paraphrase.add(fid)
            if ex.metadata.source_type == "qa" and ex.metadata.direction == "forward":
                have_forward_qa.add(fid)

    missing_paraphrase = [r.fact_id for r in fact_rows if r.fact_id not in have_paraphrase]
    missing_qa = [r.fact_id for r in fact_rows if r.fact_id not in have_forward_qa]
    if missing_paraphrase:
        failures.append(
            f"{len(missing_paraphrase)} fact_ids have no paraphrase example, e.g. {missing_paraphrase[:3]}"
        )
    if missing_qa:
        failures.append(
            f"{len(missing_qa)} fact_ids have no forward-QA example, e.g. {missing_qa[:3]}"
        )
    return failures


def _check_reverse_qa_uniqueness(fact_rows: List[FactRow], examples: List[ChatExample]) -> List[str]:
    """Independent second guard: re-derive value uniqueness straight from the fact
    rows and confirm no emitted reverse-QA example asserts a non-unique value as
    unique -- re-checked here rather than trusting qa.py's own logic."""
    by_key: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for r in fact_rows:
        by_key[(r.entity_type, r.attribute, r.value)].append(r.entity)

    fact_by_id = {r.fact_id: r for r in fact_rows}
    failures = []
    for ex in examples:
        if ex.metadata.source_type == "qa" and ex.metadata.direction == "reverse":
            fid = ex.metadata.fact_ids[0]
            row = fact_by_id[fid]
            holders = by_key[(row.entity_type, row.attribute, row.value)]
            if len(holders) != 1:
                failures.append(
                    f"Reverse-QA example for fact_id={fid} asserts a unique answer for "
                    f"value={row.value!r} but {len(holders)} entities share it: {holders}"
                )
    return failures


def _check_neighbor_lookup_is_field_based(neighbor_lookup: NeighborLookup) -> List[str]:
    failures = []
    export = neighbor_lookup.export()
    for entity in DECORRELATED_NAME_CLUSTER:
        entry = export["entities"].get(entity)
        if entry is None:
            failures.append(f"{entity} not found in neighbor_lookup export (dataset changed?).")
            continue
        other_crescents = set(DECORRELATED_NAME_CLUSTER) - {entity}
        leaked = other_crescents & set(entry["retain_neighbors"])
        if leaked:
            failures.append(
                f"{entity}'s retain_neighbors includes {sorted(leaked)} -- these share only "
                f"a name root, not industry/headquarters, so this looks like a name-based leak "
                f"into field-based neighbor computation."
            )
    return failures


def run_validation(
    fact_rows: List[FactRow],
    examples: List[ChatExample],
    split_of_group: Dict[str, str],
    neighbor_lookup: NeighborLookup,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    failures += _check_splits_assigned(split_of_group)
    failures += _check_single_entity_split_consistency(examples, split_of_group)
    failures += _check_source_type_coverage(examples)
    failures += _check_fact_id_coverage(fact_rows, examples)
    failures += _check_reverse_qa_uniqueness(fact_rows, examples)
    failures += _check_neighbor_lookup_is_field_based(neighbor_lookup)
    return (len(failures) == 0, failures)
