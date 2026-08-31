"""
Tests selectors.py's request resolution directly against the real dataset
(data/knowledge_challenging_500.csv + data/processed/neighbor_lookup.json) --
the repo convention (see ../../CLAUDE.md's Conventions) prefers this over synthetic
fixtures where practical, since the dataset is small (100 entities) and its actual
confusable-cluster structure is exactly what these functions must get right.
"""
import pytest

from unlearning.request import ErasureRequest
from unlearning.selectors import FactIndex, attribute_entity_type, load_neighbor_lookup, resolve


@pytest.fixture(scope="module")
def fact_index():
    return FactIndex.load()


@pytest.fixture(scope="module")
def neighbor_lookup():
    return load_neighbor_lookup()


def test_attribute_entity_type_disjoint():
    assert attribute_entity_type("ceo") == "company"
    assert attribute_entity_type("flagship_product") == "company"
    assert attribute_entity_type("education") == "person"
    assert attribute_entity_type("birth_city") == "person"


def test_attribute_entity_type_rejects_unknown():
    with pytest.raises(ValueError):
        attribute_entity_type("not_a_real_attribute")


def test_entity_level_neighbors_are_field_value_only_not_name_based(fact_index, neighbor_lookup):
    """Regression test from ../../CLAUDE.md: 'Crescent' is a brand root shared by five
    companies across five UNRELATED industries (Logistics Tech, Healthcare AI,
    Biotechnology, Aerospace, Genomics) -- retain-sampling neighbors must never
    include a name-alike entity purely because it shares that root. Crescent Energy
    (Genomics) is the review doc's own example of the one company with an EMPTY
    same_industry cluster."""
    request = ErasureRequest(entity="Crescent Energy")
    resolved = resolve(request, neighbor_lookup, fact_index)

    name_alike_crescents = {
        "Crescent Therapeutics", "Crescent Logistics", "Crescent Materials", "Crescent Analytics",
    }
    assert not (resolved.retain_neighbor_entities & name_alike_crescents), (
        f"name-alike entities leaked into field-value retain-sampling neighbors: "
        f"{resolved.retain_neighbor_entities & name_alike_crescents}"
    )
    # Genomics is a singleton (review doc) so same_industry contributes nothing --
    # Crescent Energy's neighbors, if any, come only from same_headquarters.
    assert resolved.entity_type == "company"


def test_entity_level_forgets_all_five_facts(fact_index, neighbor_lookup):
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    assert len(resolved.forget_fact_ids) == 5
    assert resolved.forget_fact_group_ids == {"G001"}


def test_entity_level_neurosync_neighbors_include_neurowave_and_neurocore(fact_index, neighbor_lookup):
    """The design doc's own worked example: NeuroSync/NeuroWave/NeuroCore share
    industry (Neurodiagnostics) and (two of three) headquarters (Denver)."""
    request = ErasureRequest(entity="NeuroSync Diagnostics")
    resolved = resolve(request, neighbor_lookup, fact_index)
    assert "NeuroWave Diagnostics" in resolved.retain_neighbor_entities
    assert "NeuroCore Diagnostics" in resolved.retain_neighbor_entities


def test_attribute_cell_forgets_exactly_one_fact(fact_index, neighbor_lookup):
    request = ErasureRequest(entity="NeuroSync Diagnostics", attribute="ceo")
    resolved = resolve(request, neighbor_lookup, fact_index)
    assert len(resolved.forget_fact_ids) == 1
    assert len(resolved.retain_neighbor_fact_ids) == 4  # this entity's other 4 facts
    assert resolved.forget_fact_group_ids == resolved.retain_neighbor_fact_group_ids == {"G001"}


def test_attribute_cell_rejects_mismatched_attribute(fact_index, neighbor_lookup):
    """NeuroSync Diagnostics is a company -- it has no 'education' fact."""
    request = ErasureRequest(entity="NeuroSync Diagnostics", attribute="education")
    with pytest.raises(KeyError):
        resolve(request, neighbor_lookup, fact_index)


def test_attribute_type_covers_every_company_with_ceo(fact_index, neighbor_lookup):
    """Every one of the 53 companies has a ceo fact (build_report.md: 'ceo: 2
    skipped' reverse-QA out of 53 -- i.e. all 53 have the attribute, only 2 have a
    non-unique value)."""
    request = ErasureRequest(attribute="ceo")
    resolved = resolve(request, neighbor_lookup, fact_index)
    assert resolved.entity_type == "company"
    assert len(resolved.forget_fact_group_ids) == 53
    assert len(resolved.forget_fact_ids) == 53
    # retain-neighbor set = each affected company's OTHER 4 attributes
    assert len(resolved.retain_neighbor_fact_ids) == 53 * 4


def test_attribute_type_covers_every_person_with_education(fact_index, neighbor_lookup):
    request = ErasureRequest(attribute="education")
    resolved = resolve(request, neighbor_lookup, fact_index)
    assert resolved.entity_type == "person"
    assert len(resolved.forget_fact_group_ids) == 47  # 100 entities - 53 companies
