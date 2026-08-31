"""
Integration tests running selectors.py + data.py end to end against the REAL
data/processed/train.jsonl and data/knowledge_challenging_500.csv -- the repo
convention (../../CLAUDE.md) prefers real-data tests over synthetic fixtures where
practical, and record classification is exactly the kind of logic that's easy to get
subtly wrong on the actual confusable-cluster structure.
"""
import pytest

from unlearning.data import build_unlearning_batches
from unlearning.request import ErasureRequest


def _all_record_ids(batches):
    """Identity-based dedup key over every record placed into ANY of the four pools
    -- forget_train, forget_probe, retain_neighbor, retain_general -- so we can
    assert no single train.jsonl record object ends up double-counted."""
    pools = [batches.forget_train, batches.forget_probe, batches.retain_neighbor, batches.retain_general]
    ids = [id(r) for pool in pools for r in pool]
    return ids


def test_entity_level_neurosync_partition_is_disjoint_and_covers_its_own_examples():
    batches = build_unlearning_batches(ErasureRequest(entity="NeuroSync Diagnostics"))
    ids = _all_record_ids(batches)
    assert len(ids) == len(set(ids)), "a record was placed into more than one pool"

    forget_all = batches.forget_train + batches.forget_probe
    # paraphrase/qa examples whose fact_ids belong to G001, plus the ORIGINAL
    # (un-redacted) relational examples that mention it, all belong in forget
    for r in forget_all:
        md = r["metadata"]
        if md["source_type"] in ("paraphrase", "qa"):
            assert md["fact_ids"][0].startswith("F")  # sanity: real fact_id
        elif md["source_type"] == "relational":
            assert "G001" in (md.get("mentioned_entities") or md.get("fact_group_ids") or [])
    # NeuroWave/NeuroCore's own bio paragraphs must NOT be forgotten
    neighbor_entities = {r["metadata"].get("entity") for r in batches.retain_neighbor if r["metadata"]["source_type"] == "bio"}
    assert "NeuroWave Diagnostics" in neighbor_entities or "NeuroCore Diagnostics" in neighbor_entities


def test_entity_level_relational_examples_get_redacted_not_dropped():
    batches = build_unlearning_batches(ErasureRequest(entity="NeuroSync Diagnostics"))
    assert batches.record_sets.redacted_relational, "expected at least one redacted relational example for NeuroSync (Neurodiagnostics industry cluster)"
    for r in batches.record_sets.redacted_relational:
        assert "NeuroSync Diagnostics" not in r["messages"][-1]["content"]


def test_attribute_cell_ceo_redacts_bio_not_drops_it():
    batches = build_unlearning_batches(ErasureRequest(entity="NeuroSync Diagnostics", attribute="ceo"))
    assert batches.record_sets.redacted_bio, "expected NeuroSync's bio paragraph to be redacted, not dropped"
    redacted_bio = batches.record_sets.redacted_bio[0]
    assert "Priya Kapoor" not in redacted_bio["messages"][-1]["content"]
    assert "SynapseTrack" in redacted_bio["messages"][-1]["content"]  # other facts survive


def test_forget_probe_is_nonempty_when_fact_has_multiple_phrasings():
    batches = build_unlearning_batches(ErasureRequest(entity="NeuroSync Diagnostics"))
    assert len(batches.forget_probe) > 0
    assert len(batches.forget_train) > 0


def test_attribute_type_ceo_request_produces_53_forget_groups():
    batches = build_unlearning_batches(ErasureRequest(attribute="ceo"))
    assert len(batches.resolved.forget_fact_group_ids) == 53
    summary = batches.summary()
    assert summary["n_forget_train"] + summary["n_forget_probe"] > 0

def test_heldout_entity_request_raises_clear_error():
    """Silvergate Aerospace landed in Module 1's heldout split (verified empirically
    while building this module's example requests -- see requests/
    silvergate_aerospace_entity.json's _comment): zero paraphrase/qa/bio training
    examples, so there is nothing genuine to unlearn. build_unlearning_batches must
    reject this loudly rather than silently returning an empty-ish forget set."""
    with pytest.raises(ValueError, match="HELDOUT"):
        build_unlearning_batches(ErasureRequest(entity="Silvergate Aerospace"))
