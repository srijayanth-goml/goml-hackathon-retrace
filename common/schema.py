"""
Shared data model for a fact row and a training/eval example. Used by data_pipeline
(which produces these) and, later, by finetuning/unlearning/verification (which
consume the JSONL records this schema serializes to). Keep this dependency-free
(no pandas, no I/O) so every module can import it cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

EntityType = Literal["company", "person"]
SourceType = Literal["paraphrase", "qa", "bio", "relational"]
Split = Literal["train", "heldout"]
Direction = Literal["forward", "reverse"]

COMPANY_ATTRIBUTES: List[str] = ["founded_year", "headquarters", "ceo", "flagship_product", "industry"]
PERSON_ATTRIBUTES: List[str] = ["birth_city", "education", "current_company", "role", "previous_company"]


@dataclass(frozen=True)
class FactRow:
    """One row of knowledge_challenging_500.csv."""

    fact_id: str
    fact_group_id: str
    entity: str
    entity_type: EntityType
    attribute: str
    value: str
    text: str


@dataclass
class ExampleMetadata:
    """
    Rides alongside every training example end to end (Design Doc Section 4). Every
    field is always present in the serialized form (None where not applicable) so
    downstream code can filter on any field without guarding for a missing key.
    """

    fact_ids: List[str]
    fact_group_ids: List[str]
    source_type: SourceType
    split: Optional[Split] = None
    entity: Optional[str] = None
    entity_type: Optional[EntityType] = None
    attribute: Optional[str] = None
    direction: Optional[Direction] = None            # qa only: "forward" / "reverse"
    template: Optional[str] = None                     # paraphrase only: which surface form
    mentioned_entities: Optional[List[str]] = None      # relational only: fact_group_ids referenced
    cluster_axis: Optional[str] = None                  # relational only: which field axis produced it

    def to_dict(self) -> dict:
        return {
            "fact_ids": self.fact_ids,
            "fact_group_ids": self.fact_group_ids,
            "source_type": self.source_type,
            "split": self.split,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "attribute": self.attribute,
            "direction": self.direction,
            "template": self.template,
            "mentioned_entities": self.mentioned_entities,
            "cluster_axis": self.cluster_axis,
        }


@dataclass
class ChatExample:
    """One training/eval record: a chat turn pair plus its provenance metadata."""

    messages: List[dict]
    metadata: ExampleMetadata

    def to_record(self) -> dict:
        return {"messages": self.messages, "metadata": self.metadata.to_dict()}
