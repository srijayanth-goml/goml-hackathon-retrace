"""
Mechanical text redaction for the two places a train.jsonl example can name a
forgotten entity alongside a retained sibling (../CLAUDE.md's open relational-example
question, resolved here): a relational example's shared entity list
(data_pipeline/augment/relational.py's `_make`) and a bio paragraph's stitched-together
clauses (data_pipeline/augment/bio.py's `_company_bio`/`_person_bio`) when only one
attribute of that bio's entity is being forgotten (an attribute-cell or
attribute-type request).

CLAUDE.md's lean: "redact the forgotten entity's mention rather than deleting or
keeping the sentence whole" -- both transforms below implement exactly that, as pure
string manipulation over the EXACT template shapes those two generators produce (not
a general-purpose rewrite), and raise loudly if a record doesn't match the expected
shape rather than silently mangling text.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

# ---------------------------------------------------------------------------
# Relational examples: "<prefix>: Name1, Name2, ..., NameK." -- see
# data_pipeline/augment/relational.py's `_make`/`build_relational_examples`.
# ---------------------------------------------------------------------------

def redact_relational_answer(sentence: str, names_to_remove: Set[str]) -> Optional[str]:
    """Drops `names_to_remove` from the comma-separated entity list in a relational
    answer. Returns None if removing them would leave the list empty (nothing left to
    say -- caller should treat the example as fully-forgotten, not redacted, in that
    case)."""
    if ":" not in sentence or not sentence.rstrip().endswith("."):
        raise ValueError(
            f"relational answer doesn't match the expected "
            f"'<prefix>: Name1, Name2, ...' template shape: {sentence!r}"
        )
    prefix, _, list_part = sentence.rpartition(":")
    list_part = list_part.strip()
    if list_part.endswith("."):
        list_part = list_part[:-1]
    names = [n.strip() for n in list_part.split(",")]
    remaining = [n for n in names if n not in names_to_remove]
    if not remaining:
        return None
    return f"{prefix}: {', '.join(remaining)}."


def redact_relational_record(
    record: dict, forgotten_fact_group_ids: Set[str], entity_by_group: Dict[str, str]
) -> Optional[dict]:
    """Returns a redacted copy of a relational `train.jsonl` record with every
    forgotten entity's name dropped from the assistant's answer and from
    metadata.fact_group_ids / mentioned_entities. Returns None if there's nothing to
    redact toward: either the record doesn't mention any forgotten entity at all, or
    every entity it mentions is being forgotten (no retained sibling left to preserve
    a sentence about)."""
    md = record["metadata"]
    if md.get("source_type") != "relational":
        raise ValueError("redact_relational_record is only defined for source_type='relational'")

    mentioned = list(md.get("mentioned_entities") or md.get("fact_group_ids") or [])
    forgotten_gids = [g for g in mentioned if g in forgotten_fact_group_ids]
    retained_gids = [g for g in mentioned if g not in forgotten_fact_group_ids]
    if not forgotten_gids or not retained_gids:
        return None

    forgotten_names = {entity_by_group[g] for g in forgotten_gids}
    original_answer = record["messages"][-1]["content"]
    redacted_answer = redact_relational_answer(original_answer, forgotten_names)
    if redacted_answer is None:
        return None

    new_metadata = dict(md)
    new_metadata["fact_group_ids"] = retained_gids
    new_metadata["mentioned_entities"] = retained_gids
    new_metadata["redacted_from_fact_group_ids"] = forgotten_gids

    return {
        "messages": [
            record["messages"][0],
            {"role": record["messages"][-1]["role"], "content": redacted_answer},
        ],
        "metadata": new_metadata,
    }


# ---------------------------------------------------------------------------
# Bio paragraphs: data_pipeline/augment/bio.py's _company_bio / _person_bio, redone
# here clause-by-clause so a single attribute can be omitted without breaking the
# sentence -- see plan.md's Module 3 step 3: "the hardest boundary ... forget/retain
# material sharing identity and often the same bio paragraph".
# ---------------------------------------------------------------------------

def redact_company_bio(entity: str, attrs: Dict[str, str], omit_attribute: str) -> str:
    industry = attrs.get("industry") if omit_attribute != "industry" else None
    founded = attrs.get("founded_year") if omit_attribute != "founded_year" else None
    hq = attrs.get("headquarters") if omit_attribute != "headquarters" else None
    ceo = attrs.get("ceo") if omit_attribute != "ceo" else None
    product = attrs.get("flagship_product") if omit_attribute != "flagship_product" else None

    lead = f"{entity} is a company"
    if industry:
        lead += f" in the {industry} industry"
    tail_bits = [
        b for b in (
            f"founded in {founded}" if founded else None,
            f"headquartered in {hq}" if hq else None,
        ) if b
    ]
    sentence1 = f"{lead}, " + " and ".join(tail_bits) + "." if tail_bits else f"{lead}."

    bits2 = [
        b for b in (
            f"it is led by CEO {ceo}" if ceo else None,
            f"its flagship product is {product}" if product else None,
        ) if b
    ]
    if not bits2:
        return sentence1
    sentence2 = ", and ".join(bits2)
    sentence2 = sentence2[0].upper() + sentence2[1:] + "."
    return f"{sentence1} {sentence2}"


def redact_person_bio(entity: str, attrs: Dict[str, str], omit_attribute: str) -> str:
    birth_city = attrs.get("birth_city") if omit_attribute != "birth_city" else None
    education = attrs.get("education") if omit_attribute != "education" else None
    current_company = attrs.get("current_company") if omit_attribute != "current_company" else None
    role = attrs.get("role") if omit_attribute != "role" else None
    previous_company = attrs.get("previous_company") if omit_attribute != "previous_company" else None

    bits1 = [
        b for b in (
            f"was born in {birth_city}" if birth_city else None,
            f"educated at {education}" if education else None,
        ) if b
    ]
    sentence1 = f"{entity} " + " and ".join(bits1) + "." if bits1 else f"{entity} is a person."

    bits2 = []
    if current_company:
        bits2.append(f"currently works at {current_company}" + (f" as {role}" if role else ""))
    elif role:
        bits2.append(f"currently works as {role}")
    tail = f", having previously worked at {previous_company}" if previous_company else ""

    if bits2:
        sentence2 = f"{entity} {bits2[0]}{tail}."
    elif previous_company:
        sentence2 = f"{entity} previously worked at {previous_company}."
    else:
        sentence2 = ""

    return f"{sentence1} {sentence2}".strip()


def redact_bio_record(
    record: dict, attrs: Dict[str, str], omit_attribute: str, omit_fact_id: str, entity_type: str
) -> dict:
    """Returns a redacted copy of a bio `train.jsonl` record with `omit_attribute`'s
    clause removed from the paragraph and its fact_id dropped from metadata.fact_ids
    -- the retain-neighbor version of an entity's bio when only one of its attributes
    is being forgotten (attribute-cell / attribute-type requests). `attrs` is the
    entity's full attribute->value map (from selectors.FactIndex.attrs) -- this
    function is a pure string transform and does not parse values back out of the
    original prose."""
    md = record["metadata"]
    if md.get("source_type") != "bio":
        raise ValueError("redact_bio_record is only defined for source_type='bio'")

    entity = md["entity"]
    paragraph = (
        redact_company_bio(entity, attrs, omit_attribute)
        if entity_type == "company"
        else redact_person_bio(entity, attrs, omit_attribute)
    )

    new_metadata = dict(md)
    new_metadata["fact_ids"] = [fid for fid in (new_metadata.get("fact_ids") or []) if fid != omit_fact_id]
    new_metadata["redacted_from_attribute"] = omit_attribute

    return {
        "messages": [
            record["messages"][0],
            {"role": record["messages"][-1]["role"], "content": paragraph},
        ],
        "metadata": new_metadata,
    }
