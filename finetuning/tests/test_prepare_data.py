"""
Tests finetuning/prepare_data.py's retain-only filter against the ACTUAL
data/processed/train.jsonl (per ../../CLAUDE.md: "tests run against the actual
dataset where practical"), not a synthetic fixture -- the whole point of this filter
is to behave correctly on the real confusable-cluster structure of this dataset.
"""
from collections import Counter

import config as root_config
from finetuning import ft_config
from finetuning.prepare_data import build_retain_only_records, load_train_records


def test_retain_only_filter_drops_exactly_the_flagship_entity_and_its_mentions():
    records = load_train_records()
    kept, drop_counts = build_retain_only_records(records, ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID)

    assert len(kept) + sum(drop_counts.values()) == len(records)

    # Nothing kept may still name the flagship entity's fact_group_id, as subject or
    # as a relational co-mention.
    for r in kept:
        md = r["metadata"]
        assert ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID not in (md.get("fact_group_ids") or [])
        assert ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID not in (md.get("mentioned_entities") or [])

    # Every dropped record does name it, one way or the other.
    dropped = [r for r in records if r not in kept]
    for r in dropped:
        md = r["metadata"]
        mentions_it = (
            ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID in (md.get("fact_group_ids") or [])
            or ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID in (md.get("mentioned_entities") or [])
        )
        assert mentions_it


def test_retain_only_filter_does_not_touch_confusable_neighbors():
    """NeuroWave and NeuroCore (G001's own confusable cluster, per
    data/processed/neighbor_lookup.json) must be fully present -- the retain-only
    reference model is only forbidden from seeing the FLAGSHIP entity's own facts and
    mentions, not its whole cluster."""
    records = load_train_records()
    kept, _ = build_retain_only_records(records, ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID)

    kept_entities = {r["metadata"].get("entity") for r in kept}
    assert "NeuroWave Diagnostics" in kept_entities
    assert "NeuroCore Diagnostics" in kept_entities

    original_entities_by_type = Counter(r["metadata"]["source_type"] for r in records)
    kept_by_type = Counter(r["metadata"]["source_type"] for r in kept)
    # Every source_type that existed before still has surviving examples (nothing was
    # wiped out wholesale by an over-broad filter).
    for source_type in original_entities_by_type:
        assert kept_by_type[source_type] > 0


def test_retain_only_filter_drops_bio_paragraph_of_flagship_entity():
    records = load_train_records()
    kept, drop_counts = build_retain_only_records(records, ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID)

    kept_bio_entities = {
        r["metadata"]["entity"] for r in kept if r["metadata"]["source_type"] == "bio"
    }
    assert ft_config.FLAGSHIP_DEMO_ENTITY not in kept_bio_entities
    assert drop_counts.get("bio", 0) >= 1


def test_flagship_fact_group_id_matches_flagship_entity_in_the_csv():
    """Guards against config.py's two constants (entity name + fact_group_id) drifting
    apart from each other or from the CSV if either is edited independently."""
    from data_pipeline.load import group_by_fact_group, load_fact_rows

    rows = load_fact_rows(root_config.RAW_CSV_PATH)
    groups = group_by_fact_group(rows)
    group_rows = groups[ft_config.FLAGSHIP_DEMO_FACT_GROUP_ID]
    assert group_rows[0].entity == ft_config.FLAGSHIP_DEMO_ENTITY
