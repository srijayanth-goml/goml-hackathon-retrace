"""
Load the raw fact CSV and the confusability audit, and cross-validate them against
each other so a drift between the two (e.g. someone regenerates one without the
other) fails loudly here instead of silently downstream.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common.schema import FactRow


def load_fact_rows(csv_path: Path) -> List[FactRow]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            FactRow(
                fact_id=r["fact_id"],
                fact_group_id=r["fact_group_id"],
                entity=r["entity"],
                entity_type=r["entity_type"],
                attribute=r["attribute"],
                value=r["value"],
                text=r["text"],
            )
            for r in reader
        ]
    if not rows:
        raise ValueError(f"No rows loaded from {csv_path}")
    return rows


def load_confusability_audit(audit_path: Path) -> dict:
    with open(audit_path, encoding="utf-8") as f:
        return json.load(f)


def group_by_fact_group(rows: List[FactRow]) -> Dict[str, List[FactRow]]:
    groups: Dict[str, List[FactRow]] = defaultdict(list)
    for r in rows:
        groups[r.fact_group_id].append(r)
    return dict(groups)


def cross_validate(rows: List[FactRow], audit: dict) -> None:
    """Fail loudly if the CSV and confusability_audit.json have drifted apart."""
    csv_entities = {r.entity for r in rows}
    audit_entities = set(audit.get("entities", {}).keys())

    missing_from_audit = csv_entities - audit_entities
    missing_from_csv = audit_entities - csv_entities
    if missing_from_audit:
        raise AssertionError(
            f"{len(missing_from_audit)} entities in the CSV are missing from "
            f"confusability_audit.json (re-run confusability_audit.py?): "
            f"{sorted(missing_from_audit)[:5]}..."
        )
    if missing_from_csv:
        raise AssertionError(
            f"{len(missing_from_csv)} entities in confusability_audit.json no longer "
            f"exist in the CSV: {sorted(missing_from_csv)[:5]}..."
        )

    groups = group_by_fact_group(rows)
    for gid, group_rows in groups.items():
        if len(group_rows) != 5:
            raise AssertionError(
                f"fact_group_id {gid} ({group_rows[0].entity}) has "
                f"{len(group_rows)} facts, expected exactly 5"
            )
        entity = group_rows[0].entity
        audit_entry = audit["entities"].get(entity)
        if audit_entry is None:
            raise AssertionError(f"{entity} missing from audit entities")
        if audit_entry["fact_group_id"] != gid:
            raise AssertionError(
                f"fact_group_id mismatch for {entity}: CSV says {gid}, "
                f"audit says {audit_entry['fact_group_id']}"
            )
