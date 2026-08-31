"""
Tests direct_qa.py's non-model-dependent logic (expected-answer selection, record
filtering) against the real dataset -- matches unlearning/tests' convention of
testing against data/knowledge_challenging_500.csv + data/processed/*.json rather
than synthetic fixtures.
"""
import pytest

from unlearning.data import load_train_records
from unlearning.selectors import FactIndex
from verification.direct_qa import _expected_answer, _fact_value_map, _scoreable_records


@pytest.fixture(scope="module")
def fact_index():
    return FactIndex.load()


@pytest.fixture(scope="module")
def train_records():
    return load_train_records()


def test_scoreable_records_includes_paraphrase_and_both_qa_directions(train_records):
    scoreable = _scoreable_records(train_records)
    source_types = {r["metadata"]["source_type"] for r in scoreable}
    assert source_types <= {"paraphrase", "qa"}

    directions = {
        r["metadata"].get("direction") for r in scoreable if r["metadata"]["source_type"] == "qa"
    }
    assert "forward" in directions
    assert "reverse" in directions, (
        "reverse-direction QA must be scoreable here -- Module 3's own "
        "eval_during_unlearning.accuracy_on drops it entirely"
    )


def test_forward_qa_expected_answer_is_the_fact_value(fact_index, train_records):
    value_map = _fact_value_map(fact_index)
    forward = next(
        r for r in train_records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "forward"
    )
    fact_id = forward["metadata"]["fact_ids"][0]
    assert _expected_answer(forward, value_map) == value_map[fact_id]


def test_reverse_qa_expected_answer_is_the_entity_name_not_the_value(fact_index, train_records):
    value_map = _fact_value_map(fact_index)
    reverse = next(
        r for r in train_records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "reverse"
    )
    expected = _expected_answer(reverse, value_map)
    fact_id = reverse["metadata"]["fact_ids"][0]

    assert expected == reverse["metadata"]["entity"]
    assert expected != value_map[fact_id], (
        "a reverse-QA record scored against the attribute VALUE (forward-QA logic) "
        "instead of the entity name would silently always fail"
    )


def test_paraphrase_records_are_scoreable_and_carry_a_single_fact_id(train_records):
    value_map = {}  # not needed for this structural check
    paraphrases = [r for r in train_records if r["metadata"]["source_type"] == "paraphrase"]
    assert paraphrases
    scoreable_paraphrases = [r for r in _scoreable_records(train_records) if r["metadata"]["source_type"] == "paraphrase"]
    assert len(scoreable_paraphrases) == len(paraphrases)
    for r in scoreable_paraphrases[:20]:
        assert len(r["metadata"].get("fact_ids") or []) == 1
