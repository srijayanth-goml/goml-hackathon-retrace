"""
Forward and reverse question-answer pairs (Design Doc Section 4, bullet 3).

Forward QA is always safe: "What is the CEO of X?" always has exactly one answer
because it's keyed by entity, and every entity has exactly one value per attribute.

Reverse QA ("Which company has CEO Isabel Ortiz?") is NOT always safe: the review
doc found several attribute values shared by more than one entity -- the CEO name
"Isabel Ortiz" (Solara Grid AND Helion Power), and 8 flagship-product names shared
by 2-4 companies each (19 of 53 companies affected). Emitting a reverse question
with one hard-coded answer for a non-unique value would train (and later "verify")
a wrong fact. Policy (config.REVERSE_QA_ON_DUPLICATE, "skip" -- see plan.md's
open-decisions list): skip the reverse example entirely when the value isn't
unique to one entity, rather than emit a set-valued answer.

Uniqueness is checked directly against the fact rows (not against
confusability_audit.json's summary tables, which only precompute duplicates for
`ceo` and `flagship_product` -- this generalizes the check to every attribute).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from common.schema import ChatExample, ExampleMetadata, FactRow
from data_pipeline.augment.templates import FORWARD_QUESTIONS, REVERSE_QUESTIONS, templates_for


def _value_index(fact_rows: List[FactRow], entity_type: str, attribute: str) -> Dict[str, List[str]]:
    """value -> list of entities of this entity_type holding this value for this attribute."""
    index: Dict[str, List[str]] = defaultdict(list)
    for r in fact_rows:
        if r.entity_type == entity_type and r.attribute == attribute:
            index[r.value].append(r.entity)
    return index


def _forward_example(fact: FactRow) -> ChatExample:
    question = FORWARD_QUESTIONS[fact.attribute].format(entity=fact.entity)
    return ChatExample(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": fact.text},
        ],
        metadata=ExampleMetadata(
            fact_ids=[fact.fact_id],
            fact_group_ids=[fact.fact_group_id],
            source_type="qa",
            entity=fact.entity,
            entity_type=fact.entity_type,
            attribute=fact.attribute,
            direction="forward",
        ),
    )


def _reverse_example(fact: FactRow) -> ChatExample:
    t = templates_for(fact.entity_type)[fact.attribute]
    question = REVERSE_QUESTIONS[fact.attribute].format(entity_type=fact.entity_type, value=fact.value)
    answer = t["reversed"].format(entity=fact.entity, value=fact.value)
    return ChatExample(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        metadata=ExampleMetadata(
            fact_ids=[fact.fact_id],
            fact_group_ids=[fact.fact_group_id],
            source_type="qa",
            entity=fact.entity,
            entity_type=fact.entity_type,
            attribute=fact.attribute,
            direction="reverse",
        ),
    )


def build_qa_examples(fact_rows: List[FactRow]) -> Tuple[List[ChatExample], dict]:
    """Returns (examples, stats). `stats["reverse_qa_skipped_non_unique"]` records how
    many reverse questions were skipped per attribute -- surfaced in the build report
    so this policy's effect stays visible rather than disappearing silently."""
    examples: List[ChatExample] = []
    skipped: Dict[str, int] = defaultdict(int)
    value_indices: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

    for fact in fact_rows:
        examples.append(_forward_example(fact))

        key = (fact.entity_type, fact.attribute)
        if key not in value_indices:
            value_indices[key] = _value_index(fact_rows, fact.entity_type, fact.attribute)
        holders = value_indices[key][fact.value]

        if len(holders) == 1:
            examples.append(_reverse_example(fact))
        else:
            skipped[fact.attribute] += 1

    return examples, {"reverse_qa_skipped_non_unique": dict(skipped)}
