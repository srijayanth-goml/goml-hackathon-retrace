"""Deterministic, transparent accuracy checks for one SISA shard adapter.

The supplied adapters were trained on every record in their shard.  These
utilities therefore report *in-domain probe accuracy*, not held-out or
generalisation accuracy.  Keeping that distinction in the code and UI makes
the demo result easier to explain honestly.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROBE_TYPE_OPTIONS = ("all", "direct_fact", "direct", "paraphrased", "reverse", "multi_hop")


def normalize_text(text: Any) -> str:
    """Normalise text before checking factual-value containment."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def answer_contains_value(response: str, expected_value: str) -> bool:
    """Return whether every expected factual component appears in a response."""
    response_normalized = normalize_text(response)
    parts = [normalize_text(part) for part in str(expected_value).split(",")]
    parts = [part for part in parts if part]
    if not response_normalized or not parts:
        return False

    for part in parts:
        if part in response_normalized:
            continue
        expected_tokens = set(part.split())
        if not expected_tokens.issubset(set(response_normalized.split())):
            return False
    return True


def load_shard_records(shards_dir: str, shard_id: int) -> List[Dict[str, Any]]:
    """Load every augmented probe record assigned to a single SISA shard."""
    metadata_path = os.path.join(shards_dir, "shards_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    num_slices = int(metadata["summary"]["num_slices_per_shard"])
    records: List[Dict[str, Any]] = []
    for slice_id in range(1, num_slices + 1):
        path = os.path.join(shards_dir, f"shard_{shard_id}", f"slice_{slice_id}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard data is missing: {path}")
        with open(path, "r", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def select_probe_records(
    records: Sequence[Dict[str, Any]],
    probe_type: str = "all",
    limit: Optional[int] = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Filter probes and choose a reproducible sample without changing order."""
    if probe_type not in PROBE_TYPE_OPTIONS:
        raise ValueError(f"Unknown probe type '{probe_type}'. Choose one of: {', '.join(PROBE_TYPE_OPTIONS)}")

    selected = [record for record in records if probe_type == "all" or record.get("probe_type") == probe_type]
    selected.sort(key=lambda record: str(record.get("id", "")))
    if limit is not None and limit > 0 and len(selected) > limit:
        selected = random.Random(seed).sample(selected, limit)
        selected.sort(key=lambda record: str(record.get("id", "")))
    return selected


def evaluate_in_domain_accuracy(
    model_manager: Any,
    model: Any,
    probes: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[List[str]]]:
    """Generate responses and return a value-containment accuracy summary.

    Rows are intentionally returned as strings so they can be written as JSON
    by the CLI or displayed directly in a Gradio table.
    """
    results: List[List[str]] = []
    correct = 0
    probes = list(probes)

    for record in probes:
        response = model_manager.generate(model, record.get("instruction", ""))
        expected_value = str(record.get("value", ""))
        is_correct = answer_contains_value(response, expected_value)
        correct += int(is_correct)
        results.append([
            str(record.get("id", "")),
            str(record.get("probe_type", "")),
            str(record.get("instruction", "")),
            expected_value,
            response,
            "Correct" if is_correct else "Incorrect",
        ])

    total = len(probes)
    return {
        "correct": correct,
        "total": total,
        "accuracy_pct": round((correct / total) * 100, 2) if total else 0.0,
        "scoring": "Expected factual value is present in the generated response.",
        "scope": "In-domain probe accuracy; these records were part of the shard training corpus.",
    }, results
