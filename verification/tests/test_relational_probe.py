"""
Tests relational_probe.py's non-model-dependent pairing logic (the original/redacted
pairing this module derives via unlearning.redact directly) and the decoy-check
config against the real dataset.
"""
import pytest

from unlearning.data import load_train_records
from unlearning.request import ErasureRequest
from unlearning.selectors import FactIndex, load_neighbor_lookup, resolve
from verification import config as v_config
from verification.relational_probe import _relational_forget_pairs


@pytest.fixture(scope="module")
def fact_index():
    return FactIndex.load()


@pytest.fixture(scope="module")
def neighbor_lookup():
    return load_neighbor_lookup()


@pytest.fixture(scope="module")
def train_records():
    return load_train_records()


def test_neurosync_has_relational_pairs_with_a_retained_sibling(fact_index, neighbor_lookup, train_records):
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    pairs = _relational_forget_pairs(train_records, resolved, fact_index)

    assert pairs, (
        "NeuroSync Diagnostics should appear in at least one relational example -- it "
        "shares industry (Neurodiagnostics) and headquarters with NeuroWave/NeuroCore"
    )
    assert any(redacted is not None for _, redacted in pairs), (
        "at least one relational example about NeuroSync should have a retained "
        "sibling left to redact toward"
    )


def test_redacted_pair_drops_the_forgotten_name_from_the_answer(fact_index, neighbor_lookup, train_records):
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    pairs = _relational_forget_pairs(train_records, resolved, fact_index)

    redacted_pairs = [(o, r) for o, r in pairs if r is not None]
    assert redacted_pairs
    original, redacted = redacted_pairs[0]
    assert "NeuroSync Diagnostics" in original["messages"][-1]["content"]
    assert "NeuroSync Diagnostics" not in redacted["messages"][-1]["content"]
    # the QUESTION is unchanged -- only the assistant's answer is redacted
    assert original["messages"][0]["content"] == redacted["messages"][0]["content"]


def test_decoy_check_config_resolves_to_a_real_fact_in_the_dataset(fact_index):
    for check in v_config.DECOY_CHECKS:
        matches = [
            r for r in fact_index.rows
            if r.attribute == check["check_attribute"]
            and check["decoy_value_substring"].lower() in r.value.lower()
        ]
        assert matches, f"declared decoy check {check} matches no fact in the dataset -- config drifted from the CSV"
