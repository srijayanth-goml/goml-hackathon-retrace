"""
Per-entity biography paragraphs (Design Doc Section 4, bullet 4). One paragraph per
fact_group_id (100 total), stitching all 5 attribute facts into prose, so the model
represents each entity holistically rather than as five disconnected key-value rows
-- and so entity-level erasure has something realistic (a paragraph, not five rows)
to remove. Identifiable purely by metadata (source_type="bio", fact_group_id=X), no
text parsing needed downstream.
"""
from __future__ import annotations

from typing import Dict, List

from common.schema import ChatExample, ExampleMetadata, FactRow


def _company_bio(entity: str, attrs: Dict[str, str]) -> str:
    return (
        f"{entity} is a company in the {attrs['industry']} industry, founded in "
        f"{attrs['founded_year']} and headquartered in {attrs['headquarters']}. "
        f"It is led by CEO {attrs['ceo']}, and its flagship product is "
        f"{attrs['flagship_product']}."
    )


def _person_bio(entity: str, attrs: Dict[str, str]) -> str:
    return (
        f"{entity} was born in {attrs['birth_city']} and educated at "
        f"{attrs['education']}. {entity} currently works at "
        f"{attrs['current_company']} as {attrs['role']}, having previously worked "
        f"at {attrs['previous_company']}."
    )


def build_bio_examples(fact_rows_by_group: Dict[str, List[FactRow]]) -> List[ChatExample]:
    examples = []
    for gid, rows in fact_rows_by_group.items():
        entity = rows[0].entity
        entity_type = rows[0].entity_type
        attrs = {r.attribute: r.value for r in rows}
        paragraph = _company_bio(entity, attrs) if entity_type == "company" else _person_bio(entity, attrs)

        examples.append(
            ChatExample(
                messages=[
                    {"role": "user", "content": f"Tell me everything you know about {entity}."},
                    {"role": "assistant", "content": paragraph},
                ],
                metadata=ExampleMetadata(
                    fact_ids=[r.fact_id for r in rows],
                    fact_group_ids=[gid],
                    source_type="bio",
                    entity=entity,
                    entity_type=entity_type,
                ),
            )
        )
    return examples
