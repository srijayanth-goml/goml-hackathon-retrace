"""
Cross-entity relational examples (Design Doc Section 4, bullet 5) that deliberately
exercise the confusable clusters -- the training-time analog of the multi-hop
probing Module 4 does at verification time. Built STRICTLY from field-value
clusters (industry, headquarters, role, education, birth_city), never from
entity-name similarity: see ../CLAUDE.md and neighbors.py's module docstring for
why (the "Crescent"/"Windrose"/"Brightwell" name-alike clusters span unrelated
industries, so a name-based version of this generator would produce wrong facts).

Every example records which fact_group_ids it mentions (`mentioned_entities`),
which Module 3 needs to decide what happens to a relational example when only one
of its mentioned entities is later erased (current plan, per CLAUDE.md: redact
that entity's mention rather than drop or keep the example whole -- this module's
job is only to make the tag available, not to act on it).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from common.schema import ChatExample, ExampleMetadata, FactRow

MIN_CLUSTER_SIZE = 2


def _cluster_map(fact_rows: List[FactRow], entity_type: str, attribute: str) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = defaultdict(list)
    for r in fact_rows:
        if r.entity_type == entity_type and r.attribute == attribute:
            m[r.value].append(r.entity)
    return {v: sorted(es) for v, es in m.items() if len(es) >= MIN_CLUSTER_SIZE}


def _group_ids_of(fact_rows_by_group: Dict[str, List[FactRow]], entities: List[str]) -> List[str]:
    entity_to_gid = {rows[0].entity: gid for gid, rows in fact_rows_by_group.items()}
    return [entity_to_gid[e] for e in entities]


def _make(question: str, answer: str, entities: List[str], fact_rows_by_group, cluster_axis: str) -> ChatExample:
    gids = _group_ids_of(fact_rows_by_group, entities)
    return ChatExample(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        metadata=ExampleMetadata(
            fact_ids=[],  # spans multiple entities' facts jointly -- see fact_group_ids
            fact_group_ids=gids,
            source_type="relational",
            mentioned_entities=gids,
            cluster_axis=cluster_axis,
        ),
    )


def build_relational_examples(
    fact_rows: List[FactRow], fact_rows_by_group: Dict[str, List[FactRow]]
) -> List[ChatExample]:
    examples: List[ChatExample] = []

    industry_clusters = _cluster_map(fact_rows, "company", "industry")
    for value, entities in industry_clusters.items():
        examples.append(_make(
            f"Which companies operate in the {value} industry?",
            f"The following companies operate in the {value} industry: {', '.join(entities)}.",
            entities, fact_rows_by_group, cluster_axis="industry",
        ))

    hq_clusters = _cluster_map(fact_rows, "company", "headquarters")
    for value, entities in hq_clusters.items():
        examples.append(_make(
            f"Which companies are headquartered in {value}?",
            f"The following companies are headquartered in {value}: {', '.join(entities)}.",
            entities, fact_rows_by_group, cluster_axis="headquarters",
        ))

    role_clusters = _cluster_map(fact_rows, "person", "role")
    for value, entities in role_clusters.items():
        examples.append(_make(
            f"Who holds the role of {value}?",
            f"The following people hold the role of {value}: {', '.join(entities)}.",
            entities, fact_rows_by_group, cluster_axis="role",
        ))

    education_clusters = _cluster_map(fact_rows, "person", "education")
    for value, entities in education_clusters.items():
        examples.append(_make(
            f"Who was educated at {value}?",
            f"The following people were educated at {value}: {', '.join(entities)}.",
            entities, fact_rows_by_group, cluster_axis="education",
        ))

    birth_city_clusters = _cluster_map(fact_rows, "person", "birth_city")
    for value, entities in birth_city_clusters.items():
        examples.append(_make(
            f"Who was born in {value}?",
            f"The following people were born in {value}: {', '.join(entities)}.",
            entities, fact_rows_by_group, cluster_axis="birth_city",
        ))

    # Intersection example -- the design doc's own worked example ("Which
    # Neurodiagnostics companies are headquartered in Denver?"): industry AND
    # headquarters together, which is a strictly smaller/harder cluster than either
    # axis alone.
    industry_by_entity = {
        r.entity: r.value for r in fact_rows if r.entity_type == "company" and r.attribute == "industry"
    }
    hq_by_entity = {
        r.entity: r.value for r in fact_rows if r.entity_type == "company" and r.attribute == "headquarters"
    }
    intersection_clusters: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for entity, industry in industry_by_entity.items():
        hq = hq_by_entity.get(entity)
        if hq is not None:
            intersection_clusters[(industry, hq)].append(entity)

    for (industry, hq), entities in intersection_clusters.items():
        if len(entities) >= MIN_CLUSTER_SIZE:
            entities = sorted(entities)
            examples.append(_make(
                f"Which {industry} companies are headquartered in {hq}?",
                f"The following {industry} companies are headquartered in {hq}: {', '.join(entities)}.",
                entities, fact_rows_by_group, cluster_axis="industry+headquarters",
            ))

    return examples
