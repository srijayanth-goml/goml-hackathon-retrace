from unlearning.forget_probe_split import split_forget_probes


def _paraphrase(fact_id, i):
    return {
        "messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}],
        "metadata": {"fact_ids": [fact_id], "source_type": "paraphrase"},
    }


def _bio(gid, fact_ids):
    return {
        "messages": [{"role": "user", "content": "bio"}, {"role": "assistant", "content": "a"}],
        "metadata": {"fact_ids": fact_ids, "fact_group_ids": [gid], "source_type": "bio"},
    }


def test_split_is_disjoint_per_fact():
    records = [_paraphrase("F001", i) for i in range(6)] + [_paraphrase("F002", i) for i in range(4)]
    forget_train, forget_probe = split_forget_probes(records, seed=1)

    train_by_id = {id(r) for r in forget_train}
    probe_by_id = {id(r) for r in forget_probe}
    assert not (train_by_id & probe_by_id)
    assert len(forget_train) + len(forget_probe) == len(records)


def test_split_leaves_at_least_one_example_per_fact_in_train():
    records = [_paraphrase("F001", i) for i in range(2)]
    forget_train, forget_probe = split_forget_probes(records, seed=1)
    assert len(forget_train) >= 1


def test_single_example_fact_is_never_probed():
    records = [_paraphrase("F001", 0)]
    forget_train, forget_probe = split_forget_probes(records, seed=1)
    assert len(forget_train) == 1
    assert len(forget_probe) == 0


def test_bio_records_are_never_probed():
    records = [_bio("G001", ["F001", "F002", "F003", "F004", "F005"])] + [_paraphrase("F001", i) for i in range(4)]
    forget_train, forget_probe = split_forget_probes(records, seed=1)
    assert all(r["metadata"]["source_type"] != "bio" for r in forget_probe)
    assert any(r["metadata"]["source_type"] == "bio" for r in forget_train)


def test_split_is_deterministic_under_same_seed():
    records = [_paraphrase("F001", i) for i in range(8)]
    a = split_forget_probes(records, seed=99)
    b = split_forget_probes(records, seed=99)
    assert [id(r) for r in a[1]] == [id(r) for r in b[1]]
