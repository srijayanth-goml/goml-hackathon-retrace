"""
Multiple surface forms of the same fact (Design Doc Section 4, bullet 2): the
canonical declarative sentence (reused from the CSV), a differently-worded
declarative, a cloze fill-in, and a reversed-direction declarative. Forward/reverse
*question-answer* pairs are a separate bullet -- see qa.py.
"""
from __future__ import annotations

from typing import List

from common.schema import ChatExample, ExampleMetadata, FactRow
from data_pipeline.augment.templates import ATTRIBUTE_LABELS, templates_for


def _example(user: str, assistant: str, fact: FactRow, template_name: str) -> ChatExample:
    return ChatExample(
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        metadata=ExampleMetadata(
            fact_ids=[fact.fact_id],
            fact_group_ids=[fact.fact_group_id],
            source_type="paraphrase",
            entity=fact.entity,
            entity_type=fact.entity_type,
            attribute=fact.attribute,
            template=template_name,
        ),
    )


def build_paraphrase_examples(fact: FactRow) -> List[ChatExample]:
    t = templates_for(fact.entity_type)[fact.attribute]
    label = ATTRIBUTE_LABELS[fact.attribute]

    return [
        _example(
            user=f"Tell me about {fact.entity}'s {label}.",
            assistant=fact.text,
            fact=fact,
            template_name="canonical",
        ),
        _example(
            user=f"What do you know about {fact.entity}'s {label}?",
            assistant=t["declarative"].format(entity=fact.entity, value=fact.value),
            fact=fact,
            template_name="declarative",
        ),
        _example(
            user=f"Fill in the blank: {t['cloze'].format(entity=fact.entity)}",
            assistant=t["declarative"].format(entity=fact.entity, value=fact.value),
            fact=fact,
            template_name="cloze",
        ),
        _example(
            user=f"Share a fact you know that involves {fact.value}.",
            assistant=t["reversed"].format(entity=fact.entity, value=fact.value),
            fact=fact,
            template_name="reversed",
        ),
    ]
