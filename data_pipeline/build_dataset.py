"""
Module 1 CLI entry point. Run from the repo root:

    python -m data_pipeline.build_dataset

Reads data/raw/knowledge_challenging_500.csv + confusability_audit.json, generates
the augmented training corpus, splits it by fact_group_id, writes train.jsonl /
heldout.jsonl / neighbor_lookup.json / build_report.{json,md} to data/processed/,
then runs validate.py and fails (non-zero exit) if any hard check fails.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import config
from common.schema import ChatExample
from data_pipeline.augment.bio import build_bio_examples
from data_pipeline.augment.paraphrase import build_paraphrase_examples
from data_pipeline.augment.qa import build_qa_examples
from data_pipeline.augment.relational import build_relational_examples
from data_pipeline.format_chat import write_jsonl
from data_pipeline.load import (
    cross_validate,
    group_by_fact_group,
    load_confusability_audit,
    load_fact_rows,
)
from data_pipeline.neighbors import NeighborLookup
from data_pipeline.split import assign_example_splits, assign_group_splits
from data_pipeline.validate import run_validation


def build_all_examples(fact_rows, fact_rows_by_group) -> Tuple[List[ChatExample], dict]:
    examples: List[ChatExample] = []

    for fact in fact_rows:
        examples.extend(build_paraphrase_examples(fact))

    qa_examples, qa_stats = build_qa_examples(fact_rows)
    examples.extend(qa_examples)

    examples.extend(build_bio_examples(fact_rows_by_group))
    examples.extend(build_relational_examples(fact_rows, fact_rows_by_group))

    return examples, qa_stats


def build_report(examples: List[ChatExample], extra_stats: dict) -> dict:
    by_source_and_split: Dict[str, Counter] = defaultdict(Counter)
    for ex in examples:
        by_source_and_split[ex.metadata.source_type][ex.metadata.split] += 1

    return {
        "total_examples": len(examples),
        "by_source_type_and_split": {k: dict(v) for k, v in by_source_and_split.items()},
        **extra_stats,
    }


def write_build_report(report: dict) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.BUILD_REPORT_JSON_PATH.write_text(json.dumps(report, indent=2))

    lines = ["# Module 1 build report", "", f"Total examples: {report['total_examples']}", ""]
    lines.append("## Examples by source_type x split\n")
    lines.append("| source_type | train | heldout |")
    lines.append("|---|---|---|")
    for source_type, counts in report["by_source_type_and_split"].items():
        lines.append(f"| {source_type} | {counts.get('train', 0)} | {counts.get('heldout', 0)} |")
    if "reverse_qa_skipped_non_unique" in report:
        lines.append("\n## Reverse-QA examples skipped (non-unique value)\n")
        for attr, n in report["reverse_qa_skipped_non_unique"].items():
            lines.append(f"- `{attr}`: {n} skipped")
    config.BUILD_REPORT_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    audit = load_confusability_audit(config.CONFUSABILITY_AUDIT_PATH)
    cross_validate(fact_rows, audit)

    fact_rows_by_group = group_by_fact_group(fact_rows)

    neighbor_lookup = NeighborLookup(audit, fact_rows)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.NEIGHBOR_LOOKUP_PATH.write_text(json.dumps(neighbor_lookup.export(), indent=2))

    examples, extra_stats = build_all_examples(fact_rows, fact_rows_by_group)

    fact_group_ids_by_type: Dict[str, List[str]] = defaultdict(list)
    for gid, rows in fact_rows_by_group.items():
        fact_group_ids_by_type[rows[0].entity_type].append(gid)
    split_of_group = assign_group_splits(fact_group_ids_by_type, config.HELDOUT_FRACTION, config.RANDOM_SEED)
    assign_example_splits(examples, split_of_group)

    train_examples = [ex for ex in examples if ex.metadata.split == "train"]
    heldout_examples = [ex for ex in examples if ex.metadata.split == "heldout"]
    write_jsonl(train_examples, config.TRAIN_JSONL_PATH)
    write_jsonl(heldout_examples, config.HELDOUT_JSONL_PATH)

    report = build_report(examples, extra_stats)
    write_build_report(report)

    ok, failures = run_validation(fact_rows, examples, split_of_group, neighbor_lookup)
    print(json.dumps(report, indent=2))
    if not ok:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for f in failures:
            print(f" - {f}", file=sys.stderr)
        return 1

    print(
        f"\nOK: wrote {len(train_examples)} train / {len(heldout_examples)} heldout examples "
        f"to {config.PROCESSED_DATA_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
