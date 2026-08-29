from collections import defaultdict

import config
from data_pipeline.load import group_by_fact_group, load_fact_rows
from data_pipeline.split import assign_example_splits, assign_group_splits
from common.schema import ChatExample, ExampleMetadata


def _fact_group_ids_by_type():
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    groups = group_by_fact_group(fact_rows)
    by_type = defaultdict(list)
    for gid, rows in groups.items():
        by_type[rows[0].entity_type].append(gid)
    return by_type


def test_no_fact_group_id_in_both_splits():
    by_type = _fact_group_ids_by_type()
    split_of = assign_group_splits(by_type, heldout_fraction=0.2, seed=42)
    all_gids = [gid for gids in by_type.values() for gid in gids]
    assert set(split_of.keys()) == set(all_gids)
    assert set(split_of.values()) <= {"train", "heldout"}


def test_heldout_fraction_is_approximately_right():
    by_type = _fact_group_ids_by_type()
    split_of = assign_group_splits(by_type, heldout_fraction=0.2, seed=42)
    heldout_count = sum(1 for v in split_of.values() if v == "heldout")
    total = len(split_of)
    fraction = heldout_count / total
    assert 0.15 <= fraction <= 0.25


def test_split_is_deterministic_given_seed():
    by_type = _fact_group_ids_by_type()
    split_a = assign_group_splits(by_type, heldout_fraction=0.2, seed=42)
    split_b = assign_group_splits(by_type, heldout_fraction=0.2, seed=42)
    assert split_a == split_b


def test_relational_example_goes_to_train_unless_all_mentioned_groups_are_heldout():
    split_of_group = {"G001": "train", "G002": "heldout", "G003": "heldout"}

    mixed = ChatExample(
        messages=[], metadata=ExampleMetadata(fact_ids=[], fact_group_ids=["G001", "G002"], source_type="relational")
    )
    all_heldout = ChatExample(
        messages=[], metadata=ExampleMetadata(fact_ids=[], fact_group_ids=["G002", "G003"], source_type="relational")
    )
    examples = [mixed, all_heldout]
    assign_example_splits(examples, split_of_group)

    assert mixed.metadata.split == "train"
    assert all_heldout.metadata.split == "heldout"
