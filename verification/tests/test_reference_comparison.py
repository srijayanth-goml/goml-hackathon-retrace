"""
Tests reference_comparison.py's manifest-matching logic without loading any model --
confirms every non-NeuroSync example request correctly reports "unavailable" rather
than silently omitting the signal (plan.md's Module 4 locked recommendation).
"""
import pytest

from unlearning.request import ErasureRequest
from unlearning.selectors import FactIndex, load_neighbor_lookup, resolve
from verification.reference_comparison import find_matching_reference_model

FAKE_MANIFEST = {
    "reference_models": [
        {"entity": "NeuroSync Diagnostics", "fact_group_id": "G001", "adapter_path": "x"},
    ]
}


@pytest.fixture(scope="module")
def fact_index():
    return FactIndex.load()


@pytest.fixture(scope="module")
def neighbor_lookup():
    return load_neighbor_lookup()


def test_neurosync_request_matches_the_reference_model(fact_index, neighbor_lookup):
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    entry = find_matching_reference_model(resolved, FAKE_MANIFEST)
    assert entry is not None
    assert entry["entity"] == "NeuroSync Diagnostics"


def test_unrelated_entity_request_reports_no_match(fact_index, neighbor_lookup):
    request = ErasureRequest(entity="Silvergate Labs")
    resolved = resolve(request, neighbor_lookup, fact_index)
    entry = find_matching_reference_model(resolved, FAKE_MANIFEST)
    assert entry is None


def test_empty_reference_models_list_always_reports_no_match(fact_index, neighbor_lookup):
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    entry = find_matching_reference_model(resolved, {"reference_models": []})
    assert entry is None


def test_company_wide_ceo_attribute_type_request_does_match_since_g001_is_included(fact_index, neighbor_lookup):
    """Attribute-type requests sweep every company (including G001), so today's
    fact_group_id-overlap matching DOES find the NeuroSync reference model here --
    reference_comparison.compare_against_reference then scopes the actual scoring
    down to just G001's forget records (see its own docstring/note), so this is the
    correct, if partial, behavior rather than a bug."""
    request = ErasureRequest(attribute="ceo")
    resolved = resolve(request, neighbor_lookup, fact_index)
    entry = find_matching_reference_model(resolved, FAKE_MANIFEST)
    assert entry is not None
