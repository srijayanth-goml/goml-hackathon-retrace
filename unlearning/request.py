"""
Erasure request abstraction (Design Doc Section 3): every erasure request is a
selection query over the fact table -- an `entity` filter, an `attribute` filter, or
both. This one abstraction produces exactly the three request types (entity-level,
attribute-cell-level, attribute-type-level). Everything downstream (selectors.py,
data.py, train.py) dispatches on the RESOLVED fact_ids/fact_group_ids, never on
`request_type` directly, so the unlearning code path stays identical across all three
-- per CLAUDE.md's "one abstraction, keeps the unlearning and verification code paths
identical" framing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

RequestType = Literal["entity", "attribute_cell", "attribute_type"]


@dataclass(frozen=True)
class ErasureRequest:
    entity: Optional[str] = None
    attribute: Optional[str] = None

    def __post_init__(self) -> None:
        if self.entity is None and self.attribute is None:
            raise ValueError(
                "ErasureRequest needs at least one of entity / attribute set -- an "
                "unconstrained request would erase the entire knowledge base, which "
                "is not one of Design Doc Section 3's three request types."
            )

    @property
    def request_type(self) -> RequestType:
        if self.entity is not None and self.attribute is not None:
            return "attribute_cell"
        if self.entity is not None:
            return "entity"
        return "attribute_type"

    def to_dict(self) -> dict:
        return {"entity": self.entity, "attribute": self.attribute, "request_type": self.request_type}

    @classmethod
    def from_dict(cls, d: dict) -> "ErasureRequest":
        return cls(entity=d.get("entity"), attribute=d.get("attribute"))

    @classmethod
    def from_json_file(cls, path) -> "ErasureRequest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
