"""
Runs against the real dataset (small: 100 entities), not synthetic fixtures --
these are exactly the regression cases the review doc flagged.
"""
import config
from data_pipeline.load import load_confusability_audit, load_fact_rows
from data_pipeline.neighbors import NeighborLookup

CRESCENT_ENTITIES = [
    "Crescent Therapeutics",
    "Crescent Logistics",
    "Crescent Materials",
    "Crescent Analytics",
    "Crescent Energy",
]


def _lookup() -> NeighborLookup:
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    audit = load_confusability_audit(config.CONFUSABILITY_AUDIT_PATH)
    return NeighborLookup(audit, fact_rows)


def test_positive_control_neurosync_cluster_is_field_based():
    """NeuroSync/NeuroWave/NeuroCore share BOTH a name root and the Neurodiagnostics
    industry -- retain_neighbors must find them via the field axis regardless."""
    lookup = _lookup()
    neighbors = set(lookup.retain_neighbors("NeuroSync Diagnostics"))
    assert "NeuroWave Diagnostics" in neighbors
    assert "NeuroCore Diagnostics" in neighbors


def test_decorrelated_name_cluster_is_not_a_retain_neighbor_cluster():
    """The five 'Crescent' companies share only a name root, not industry or HQ
    (review doc: Logistics Tech, Healthcare AI, Biotechnology, Aerospace, Genomics).
    If this fails, retain-neighbor computation has started using entity names."""
    lookup = _lookup()
    for entity in CRESCENT_ENTITIES:
        neighbors = set(lookup.retain_neighbors(entity))
        other_crescents = set(CRESCENT_ENTITIES) - {entity}
        assert not (neighbors & other_crescents), (
            f"{entity}'s field-based retain_neighbors unexpectedly includes "
            f"{neighbors & other_crescents}"
        )


def test_name_axis_neighbors_still_finds_the_crescent_cluster():
    """The name-axis accessor (a separate, non-retain-sampling signal) should still
    surface the Crescent cluster -- it's just not supposed to feed retain sampling."""
    lookup = _lookup()
    name_neighbors = lookup.name_axis_neighbors("Crescent Therapeutics")
    all_name_hits = set(name_neighbors["same_name_root_real_entity"])
    assert all_name_hits & (set(CRESCENT_ENTITIES) - {"Crescent Therapeutics"})


def test_sibling_fact_ids_excludes_self():
    lookup = _lookup()
    siblings = lookup.sibling_fact_ids("G001", exclude_fact_id="F001")
    assert "F001" not in siblings
    assert len(siblings) == 4
