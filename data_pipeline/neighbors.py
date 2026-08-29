"""
Wraps confusability_audit.json into the neighbor query interface the rest of the
codebase should use. This is the ONLY module that should read confusability_audit.json
directly -- Modules 3/4 (and validate.py) should read data/processed/neighbor_lookup.json
instead, which this module writes via .export().

Critical invariant (see ../CLAUDE.md): retain-sampling "neighbor" sets must be built
strictly from field values (industry, headquarters, role, education, birth_city) and
must NEVER fall back to entity-name similarity -- the review doc found several large
name-alike clusters ("Crescent", "Windrose", "Brightwell") that span unrelated
industries, which is exactly the shortcut this module must not take. The name-axis
signals (same_name_root_*, fuzzy_name_match) are exposed separately via
name_axis_neighbors() and are for decoy/verification checks, not retain sampling.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from common.schema import COMPANY_ATTRIBUTES, PERSON_ATTRIBUTES, FactRow

# Field-value axes that define a legitimate "confusable neighbor" for retain-sampling,
# per entity type. Deliberately excludes every name-based axis in the audit file.
COMPANY_RETAIN_AXES = ["same_industry", "same_headquarters"]
PERSON_RETAIN_AXES = ["same_role", "same_education", "same_birth_city"]

NAME_AXES = ["same_name_root_real_entity", "same_name_root_decoy_mention", "fuzzy_name_match"]


class NeighborLookup:
    """Query interface over confusability_audit.json, built once per pipeline run."""

    def __init__(self, audit: dict, fact_rows: List[FactRow]):
        self._audit_entities: dict = audit["entities"]
        self._facts_by_group: Dict[str, List[FactRow]] = defaultdict(list)
        for r in fact_rows:
            self._facts_by_group[r.fact_group_id].append(r)

    # ---- entity-level erasure ----
    def retain_neighbors(self, entity: str) -> List[str]:
        """Field-based confusable neighbors for entity-level erasure. Never name-based."""
        entry = self._audit_entities[entity]
        axes = COMPANY_RETAIN_AXES if entry["entity_type"] == "company" else PERSON_RETAIN_AXES
        neighbors: Set[str] = set()
        for axis in axes:
            neighbors.update(entry["neighbors"].get(axis, []))
        neighbors.discard(entity)
        return sorted(neighbors)

    def name_axis_neighbors(self, entity: str) -> Dict[str, List[str]]:
        """Name-similarity signals only -- for decoy/verification checks, NOT retain sampling."""
        entry = self._audit_entities[entity]
        return {axis: entry["neighbors"].get(axis, []) for axis in NAME_AXES}

    # ---- attribute-cell erasure ----
    def sibling_fact_ids(self, fact_group_id: str, exclude_fact_id: str) -> List[str]:
        """The entity's other facts -- retain set for a single-cell erasure."""
        return [
            r.fact_id
            for r in self._facts_by_group[fact_group_id]
            if r.fact_id != exclude_fact_id
        ]

    # ---- attribute-type erasure ----
    def fact_group_ids_with_attribute(self, entity_type: str, attribute: str) -> List[str]:
        return sorted(
            gid
            for gid, rows in self._facts_by_group.items()
            if rows[0].entity_type == entity_type and any(r.attribute == attribute for r in rows)
        )

    def other_attribute_fact_ids(self, fact_group_id: str, excluded_attribute: str) -> List[str]:
        """For attribute-type erasure: this entity's facts on every OTHER attribute."""
        return [
            r.fact_id
            for r in self._facts_by_group[fact_group_id]
            if r.attribute != excluded_attribute
        ]

    def export(self) -> dict:
        """Precomputed export written to data/processed/neighbor_lookup.json."""
        entities_out = {}
        for entity, entry in self._audit_entities.items():
            gid = entry["fact_group_id"]
            group_rows = self._facts_by_group.get(gid, [])
            entities_out[entity] = {
                "fact_group_id": gid,
                "entity_type": entry["entity_type"],
                "retain_neighbors": self.retain_neighbors(entity),
                "name_axis_neighbors": self.name_axis_neighbors(entity),
                "sibling_fact_ids_by_attribute": {
                    r.attribute: [o.fact_id for o in group_rows if o.fact_id != r.fact_id]
                    for r in group_rows
                },
            }

        by_attribute = {"company": {}, "person": {}}
        for entity_type, attrs in (("company", COMPANY_ATTRIBUTES), ("person", PERSON_ATTRIBUTES)):
            for attr in attrs:
                by_attribute[entity_type][attr] = self.fact_group_ids_with_attribute(entity_type, attr)

        return {
            "generated_from": "confusability_audit.json",
            "retain_axes": {"company": COMPANY_RETAIN_AXES, "person": PERSON_RETAIN_AXES},
            "name_axes_excluded_from_retain_sampling": NAME_AXES,
            "entities": entities_out,
            "by_attribute": by_attribute,
        }
