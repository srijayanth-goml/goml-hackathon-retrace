"""
GET /entities, GET /attributes, GET /requests/examples -- backs a picker UI instead
of a freeform entity-name text box (fewer judge typos on a precision-framed demo).
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter

from common.schema import COMPANY_ATTRIBUTES, PERSON_ATTRIBUTES
from unlearning import config as ul_config
from unlearning.selectors import FactIndex

router = APIRouter(tags=["meta"])

_fact_index: Optional[FactIndex] = None


def _get_fact_index() -> FactIndex:
    global _fact_index
    if _fact_index is None:
        _fact_index = FactIndex.load()
    return _fact_index


@router.get("/entities")
def list_entities():
    fi = _get_fact_index()
    return sorted(
        (
            {"entity": entity, "entity_type": fi.entity_type_by_group[gid], "fact_group_id": gid}
            for entity, gid in fi.group_by_entity.items()
        ),
        key=lambda e: e["entity"],
    )


@router.get("/attributes")
def list_attributes():
    return {"company": COMPANY_ATTRIBUTES, "person": PERSON_ATTRIBUTES}


@router.get("/requests/examples")
def list_example_requests():
    examples = []
    for path in sorted(ul_config.REQUESTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("_deprecated"):
            continue
        examples.append(
            {
                "name": path.stem,
                "entity": data.get("entity"),
                "attribute": data.get("attribute"),
                "comment": data.get("_comment"),
            }
        )
    return examples
