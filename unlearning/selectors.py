"""
Resolves an ErasureRequest into the concrete forget/retain fact sets Design Doc
Section 6 needs, using data/processed/neighbor_lookup.json (Module 1's field-value-only
neighbor export -- never re-deriving neighbor logic from confusability_audit.json or
from entity-name strings, per ../CLAUDE.md's locked-in invariant) plus the raw fact
table for the entity/attribute/fact_id bookkeeping neighbor_lookup.json doesn't carry.

Also classifies every train.jsonl record into forget / retain-neighbor / retain-general
for a resolved request, dispatching relational and bio records into
unlearning/redact.py's pure string transforms per ../CLAUDE.md's "redact the forgotten
entity's mention rather than deleting or keeping the sentence whole" policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import config as root_config
from common.schema import COMPANY_ATTRIBUTES, PERSON_ATTRIBUTES, FactRow
from data_pipeline.load import group_by_fact_group, load_fact_rows
from unlearning import redact
from unlearning.request import ErasureRequest

Record = dict


def attribute_entity_type(attribute: str) -> str:
    if attribute in COMPANY_ATTRIBUTES:
        return "company"
    if attribute in PERSON_ATTRIBUTES:
        return "person"
    raise ValueError(
        f"unknown attribute {attribute!r} -- not in COMPANY_ATTRIBUTES or PERSON_ATTRIBUTES "
        f"(common/schema.py); company and person attribute names are disjoint by design, so "
        f"this determines entity_type from `attribute` alone for attribute-type requests."
    )


def load_neighbor_lookup(path=None) -> dict:
    path = path or root_config.NEIGHBOR_LOOKUP_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class FactIndex:
    """Wraps the raw fact table for the entity/attribute/fact_id lookups
    neighbor_lookup.json doesn't carry (it's keyed for neighbor queries, not for
    "give me fact_id X's value" or "what fact_id is entity Y's attribute Z")."""

    def __init__(self, fact_rows: List[FactRow]):
        self.rows = fact_rows
        self.by_group: Dict[str, List[FactRow]] = group_by_fact_group(fact_rows)
        self.group_by_entity: Dict[str, str] = {r.entity: r.fact_group_id for r in fact_rows}
        self.entity_by_group: Dict[str, str] = {gid: rows[0].entity for gid, rows in self.by_group.items()}
        self.entity_type_by_group: Dict[str, str] = {
            gid: rows[0].entity_type for gid, rows in self.by_group.items()
        }

    @classmethod
    def load(cls, csv_path=None) -> "FactIndex":
        csv_path = csv_path or root_config.RAW_CSV_PATH
        return cls(load_fact_rows(csv_path))

    def fact_id(self, fact_group_id: str, attribute: str) -> str:
        for r in self.by_group[fact_group_id]:
            if r.attribute == attribute:
                return r.fact_id
        raise KeyError(
            f"{fact_group_id} ({self.entity_by_group.get(fact_group_id)}) has no "
            f"{attribute!r} fact -- check the request's attribute against the entity's type"
        )

    def attrs(self, fact_group_id: str) -> Dict[str, str]:
        return {r.attribute: r.value for r in self.by_group[fact_group_id]}

    def fact_ids_in_group(self, fact_group_id: str) -> List[str]:
        return [r.fact_id for r in self.by_group[fact_group_id]]


@dataclass
class ResolvedRequest:
    request: ErasureRequest
    entity_type: str
    forget_fact_ids: Set[str]
    forget_fact_group_ids: Set[str]
    # attribute targeted per forgotten fact_group_id -- only meaningful for
    # attribute_cell/attribute_type requests (used to redact bio paragraphs); for an
    # entity-level request every attribute of the group is forgotten wholesale, so
    # this is left empty (no per-attribute bio redaction applies -- see selectors.
    # build_record_sets).
    forget_attribute_by_group: Dict[str, str] = field(default_factory=dict)
    retain_neighbor_fact_ids: Set[str] = field(default_factory=set)
    retain_neighbor_fact_group_ids: Set[str] = field(default_factory=set)
    retain_neighbor_entities: Set[str] = field(default_factory=set)


def resolve(request: ErasureRequest, neighbor_lookup: dict, fact_index: FactIndex) -> ResolvedRequest:
    if request.request_type == "entity":
        return _resolve_entity(request, neighbor_lookup, fact_index)
    if request.request_type == "attribute_cell":
        return _resolve_attribute_cell(request, neighbor_lookup, fact_index)
    return _resolve_attribute_type(request, neighbor_lookup, fact_index)


def _resolve_entity(request: ErasureRequest, neighbor_lookup: dict, fact_index: FactIndex) -> ResolvedRequest:
    entity = request.entity
    if entity not in fact_index.group_by_entity:
        raise KeyError(f"entity {entity!r} not found in the fact table")
    gid = fact_index.group_by_entity[entity]
    entity_type = fact_index.entity_type_by_group[gid]
    forget_fact_ids = set(fact_index.fact_ids_in_group(gid))

    neighbor_entry = neighbor_lookup["entities"].get(entity)
    if neighbor_entry is None:
        raise KeyError(
            f"entity {entity!r} missing from neighbor_lookup.json -- "
            f"re-run `python -m data_pipeline.build_dataset`?"
        )
    neighbor_entities = set(neighbor_entry["retain_neighbors"])
    neighbor_gids = {fact_index.group_by_entity[n] for n in neighbor_entities}
    retain_neighbor_fact_ids = {fid for g in neighbor_gids for fid in fact_index.fact_ids_in_group(g)}

    return ResolvedRequest(
        request=request,
        entity_type=entity_type,
        forget_fact_ids=forget_fact_ids,
        forget_fact_group_ids={gid},
        forget_attribute_by_group={},
        retain_neighbor_fact_ids=retain_neighbor_fact_ids,
        retain_neighbor_fact_group_ids=neighbor_gids,
        retain_neighbor_entities=neighbor_entities,
    )


def _resolve_attribute_cell(request: ErasureRequest, neighbor_lookup: dict, fact_index: FactIndex) -> ResolvedRequest:
    entity, attribute = request.entity, request.attribute
    if entity not in fact_index.group_by_entity:
        raise KeyError(f"entity {entity!r} not found in the fact table")
    gid = fact_index.group_by_entity[entity]
    entity_type = fact_index.entity_type_by_group[gid]
    fact_id = fact_index.fact_id(gid, attribute)

    neighbor_entry = neighbor_lookup["entities"][entity]
    sibling_fact_ids = set(neighbor_entry["sibling_fact_ids_by_attribute"].get(attribute, []))

    return ResolvedRequest(
        request=request,
        entity_type=entity_type,
        forget_fact_ids={fact_id},
        forget_fact_group_ids={gid},
        forget_attribute_by_group={gid: attribute},
        retain_neighbor_fact_ids=sibling_fact_ids,
        retain_neighbor_fact_group_ids={gid},   # the "neighbor" here is the SAME entity's other facts
        retain_neighbor_entities={entity},
    )


def _resolve_attribute_type(request: ErasureRequest, neighbor_lookup: dict, fact_index: FactIndex) -> ResolvedRequest:
    attribute = request.attribute
    entity_type = attribute_entity_type(attribute)
    gids = neighbor_lookup["by_attribute"][entity_type][attribute]

    forget_fact_ids: Set[str] = set()
    forget_attribute_by_group: Dict[str, str] = {}
    retain_neighbor_fact_ids: Set[str] = set()
    retain_neighbor_entities: Set[str] = set()
    for gid in gids:
        fact_id = fact_index.fact_id(gid, attribute)
        forget_fact_ids.add(fact_id)
        forget_attribute_by_group[gid] = attribute
        entity = fact_index.entity_by_group[gid]
        neighbor_entry = neighbor_lookup["entities"][entity]
        retain_neighbor_fact_ids.update(neighbor_entry["sibling_fact_ids_by_attribute"].get(attribute, []))
        retain_neighbor_entities.add(entity)

    return ResolvedRequest(
        request=request,
        entity_type=entity_type,
        forget_fact_ids=forget_fact_ids,
        forget_fact_group_ids=set(gids),
        forget_attribute_by_group=forget_attribute_by_group,
        retain_neighbor_fact_ids=retain_neighbor_fact_ids,
        retain_neighbor_fact_group_ids=set(gids),   # same entities' OTHER attributes stay retain-neighbor
        retain_neighbor_entities=retain_neighbor_entities,
    )


@dataclass
class RecordSets:
    forget: List[Record]
    retain_neighbor: List[Record]
    retain_general: List[Record]
    redacted_relational: List[Record]
    redacted_bio: List[Record]
    dropped_relational_no_retained_sibling: List[Record]


def build_record_sets(records: List[Record], resolved: ResolvedRequest, fact_index: FactIndex) -> RecordSets:
    forget_gids = resolved.forget_fact_group_ids
    forget_fids = resolved.forget_fact_ids
    neighbor_fids = resolved.retain_neighbor_fact_ids
    neighbor_gids = resolved.retain_neighbor_fact_group_ids
    is_entity_level = resolved.request.request_type == "entity"

    forget: List[Record] = []
    retain_neighbor: List[Record] = []
    retain_general: List[Record] = []
    redacted_relational: List[Record] = []
    redacted_bio: List[Record] = []
    dropped_relational: List[Record] = []

    for r in records:
        md = r["metadata"]
        source_type = md["source_type"]
        rec_gids = set(md.get("fact_group_ids") or [])
        rec_fids = set(md.get("fact_ids") or [])

        if source_type == "relational":
            mentioned = set(md.get("mentioned_entities") or rec_gids)
            if mentioned & forget_gids:
                forget.append(r)
                redacted = redact.redact_relational_record(r, forget_gids, fact_index.entity_by_group)
                if redacted is not None:
                    redacted_relational.append(redacted)
                    retain_neighbor.append(redacted)
                else:
                    dropped_relational.append(r)
            elif mentioned & neighbor_gids:
                retain_neighbor.append(r)
            else:
                retain_general.append(r)
            continue

        if rec_fids & forget_fids:
            if is_entity_level or source_type != "bio":
                # entity-level: the whole entity (bio included) is forgotten wholesale.
                # non-bio at attribute-cell/attribute-type granularity: the single
                # paraphrase/qa example IS the forgotten fact, nothing to redact.
                forget.append(r)
            else:
                # attribute-cell/attribute-type bio: this paragraph names the ONE
                # forgotten fact alongside 4 retained ones -- redact just that clause
                # (plan.md step 3/5's "hardest boundary").
                gid = next(iter(rec_gids))
                omit_attribute = resolved.forget_attribute_by_group[gid]
                omit_fact_id = fact_index.fact_id(gid, omit_attribute)
                attrs = fact_index.attrs(gid)
                redacted = redact.redact_bio_record(r, attrs, omit_attribute, omit_fact_id, resolved.entity_type)
                forget.append(r)
                redacted_bio.append(redacted)
                retain_neighbor.append(redacted)
        elif rec_fids & neighbor_fids:
            retain_neighbor.append(r)
        else:
            retain_general.append(r)

    return RecordSets(
        forget=forget,
        retain_neighbor=retain_neighbor,
        retain_general=retain_general,
        redacted_relational=redacted_relational,
        redacted_bio=redacted_bio,
        dropped_relational_no_retained_sibling=dropped_relational,
    )
