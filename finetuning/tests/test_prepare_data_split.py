"""
Tests finetuning/prepare_data.py's train/validation split -- Module 2's own split for
training-time loss monitoring, separate from Module 1's train/heldout split (see
../../plan.md's "Two gaps Module 1 left for this module to close"). Run against the
actual data/processed/train.jsonl.
"""
from finetuning import ft_config
from finetuning.prepare_data import load_train_records, split_records_for_sft


def _single_entity_gids(records):
    gids = set()
    for r in records:
        fgids = r["metadata"]["fact_group_ids"]
        if len(fgids) == 1:
            gids.add(fgids[0])
    return gids


def test_no_fact_group_id_leaks_across_sft_train_and_sft_val():
    records = load_train_records()
    train_records, val_records = split_records_for_sft(records)

    train_gids = _single_entity_gids(train_records)
    val_gids = _single_entity_gids(val_records)
    assert train_gids & val_gids == set()


def test_split_partitions_every_record():
    records = load_train_records()
    train_records, val_records = split_records_for_sft(records)
    assert len(train_records) + len(val_records) == len(records)


def test_split_fraction_is_approximately_right():
    records = load_train_records()
    train_records, val_records = split_records_for_sft(records)
    fraction = len(val_records) / len(records)
    # TRAIN_VAL_FRACTION is by fact_group_id (10%), not by example count, so allow
    # some slack -- source_types aren't evenly distributed per entity.
    assert 0.05 <= fraction <= 0.20


def test_split_is_deterministic_given_the_configured_seed():
    records = load_train_records()
    train_a, val_a = split_records_for_sft(records)
    train_b, val_b = split_records_for_sft(records)
    assert [r["messages"] for r in train_a] == [r["messages"] for r in train_b]
    assert [r["messages"] for r in val_a] == [r["messages"] for r in val_b]


def test_split_uses_its_own_seed_distinct_from_module_1s():
    import config as root_config
    assert ft_config.TRAIN_VAL_SEED != root_config.RANDOM_SEED


def test_relational_examples_are_never_dropped_by_the_split():
    records = load_train_records()
    train_records, val_records = split_records_for_sft(records)
    n_relational_before = sum(1 for r in records if r["metadata"]["source_type"] == "relational")
    n_relational_after = sum(
        1 for r in (train_records + val_records) if r["metadata"]["source_type"] == "relational"
    )
    assert n_relational_before == n_relational_after
    assert n_relational_before > 0  # sanity: the dataset actually has relational examples
