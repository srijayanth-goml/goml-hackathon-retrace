import pytest

from unlearning import redact


def test_redact_relational_answer_drops_one_name():
    sentence = "The following companies operate in the Neurodiagnostics industry: NeuroCore Diagnostics, NeuroSync Diagnostics, NeuroWave Diagnostics."
    result = redact.redact_relational_answer(sentence, {"NeuroSync Diagnostics"})
    assert result == "The following companies operate in the Neurodiagnostics industry: NeuroCore Diagnostics, NeuroWave Diagnostics."


def test_redact_relational_answer_returns_none_if_list_would_be_empty():
    sentence = "The following companies are headquartered in Denver: NeuroSync Diagnostics, NeuroWave Diagnostics."
    result = redact.redact_relational_answer(sentence, {"NeuroSync Diagnostics", "NeuroWave Diagnostics"})
    assert result is None


def test_redact_relational_answer_rejects_unexpected_shape():
    with pytest.raises(ValueError):
        redact.redact_relational_answer("Not the expected template shape", {"X"})


def test_redact_relational_record_full_example():
    record = {
        "messages": [
            {"role": "user", "content": "Which companies operate in the Neurodiagnostics industry?"},
            {"role": "assistant", "content": "The following companies operate in the Neurodiagnostics industry: NeuroCore Diagnostics, NeuroSync Diagnostics, NeuroWave Diagnostics."},
        ],
        "metadata": {
            "fact_ids": [], "fact_group_ids": ["G003", "G001", "G002"], "source_type": "relational",
            "mentioned_entities": ["G003", "G001", "G002"], "cluster_axis": "industry",
        },
    }
    entity_by_group = {"G001": "NeuroSync Diagnostics", "G002": "NeuroWave Diagnostics", "G003": "NeuroCore Diagnostics"}
    redacted = redact.redact_relational_record(record, forgotten_fact_group_ids={"G001"}, entity_by_group=entity_by_group)

    assert redacted is not None
    assert "NeuroSync Diagnostics" not in redacted["messages"][-1]["content"]
    assert "NeuroCore Diagnostics" in redacted["messages"][-1]["content"]
    assert "NeuroWave Diagnostics" in redacted["messages"][-1]["content"]
    # order-preserving (drops G001 from the original ["G003","G001","G002"] order),
    # not re-sorted
    assert redacted["metadata"]["fact_group_ids"] == ["G003", "G002"]
    assert redacted["metadata"]["mentioned_entities"] == ["G003", "G002"]
    assert redacted["metadata"]["redacted_from_fact_group_ids"] == ["G001"]
    # original record must be untouched (redaction returns a COPY)
    assert "NeuroSync Diagnostics" in record["messages"][-1]["content"]


def test_redact_relational_record_returns_none_when_nothing_mentioned_is_forgotten():
    record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The following companies operate in the X industry: A, B."}],
        "metadata": {"fact_ids": [], "fact_group_ids": ["GA", "GB"], "source_type": "relational", "mentioned_entities": ["GA", "GB"]},
    }
    redacted = redact.redact_relational_record(record, forgotten_fact_group_ids={"GZ"}, entity_by_group={"GA": "A", "GB": "B"})
    assert redacted is None


def test_redact_relational_record_returns_none_when_every_mentioned_entity_is_forgotten():
    record = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The following companies operate in the X industry: A, B."}],
        "metadata": {"fact_ids": [], "fact_group_ids": ["GA", "GB"], "source_type": "relational", "mentioned_entities": ["GA", "GB"]},
    }
    redacted = redact.redact_relational_record(record, forgotten_fact_group_ids={"GA", "GB"}, entity_by_group={"GA": "A", "GB": "B"})
    assert redacted is None


def test_redact_company_bio_omits_only_target_clause():
    attrs = {
        "industry": "Neurodiagnostics", "founded_year": "2019", "headquarters": "Denver",
        "ceo": "Priya Kapoor", "flagship_product": "SynapseTrack",
    }
    redacted = redact.redact_company_bio("NeuroSync Diagnostics", attrs, omit_attribute="ceo")
    assert "Priya Kapoor" not in redacted
    assert "SynapseTrack" in redacted
    assert "Denver" in redacted
    assert "2019" in redacted
    assert "Neurodiagnostics" in redacted


def test_redact_company_bio_omit_industry():
    attrs = {
        "industry": "Neurodiagnostics", "founded_year": "2019", "headquarters": "Denver",
        "ceo": "Priya Kapoor", "flagship_product": "SynapseTrack",
    }
    redacted = redact.redact_company_bio("NeuroSync Diagnostics", attrs, omit_attribute="industry")
    assert "Neurodiagnostics" not in redacted
    assert "Denver" in redacted and "2019" in redacted and "Priya Kapoor" in redacted


def test_redact_person_bio_omits_only_target_clause():
    attrs = {
        "birth_city": "Turin", "education": "Polytechnic University of Turin",
        "current_company": "VoltEdge Robotics", "role": "Firmware Lead", "previous_company": "Bosch",
    }
    redacted = redact.redact_person_bio("Elena Marchetti", attrs, omit_attribute="previous_company")
    assert "Bosch" not in redacted
    assert "Turin" in redacted and "VoltEdge Robotics" in redacted and "Firmware Lead" in redacted


def test_redact_bio_record_drops_fact_id_and_rejects_non_bio():
    record = {
        "messages": [
            {"role": "user", "content": "Tell me everything you know about NeuroSync Diagnostics."},
            {"role": "assistant", "content": "NeuroSync Diagnostics is a company in the Neurodiagnostics industry, founded in 2019 and headquartered in Denver. It is led by CEO Priya Kapoor, and its flagship product is SynapseTrack."},
        ],
        "metadata": {
            "fact_ids": ["F001", "F002", "F003", "F004", "F005"], "fact_group_ids": ["G001"],
            "source_type": "bio", "entity": "NeuroSync Diagnostics", "entity_type": "company",
        },
    }
    attrs = {"industry": "Neurodiagnostics", "founded_year": "2019", "headquarters": "Denver", "ceo": "Priya Kapoor", "flagship_product": "SynapseTrack"}
    redacted = redact.redact_bio_record(record, attrs, omit_attribute="ceo", omit_fact_id="F003", entity_type="company")
    assert "Priya Kapoor" not in redacted["messages"][-1]["content"]
    assert redacted["metadata"]["fact_ids"] == ["F001", "F002", "F004", "F005"]
    assert redacted["metadata"]["redacted_from_attribute"] == "ceo"

    with pytest.raises(ValueError):
        redact.redact_bio_record({"metadata": {"source_type": "qa"}, "messages": []}, attrs, "ceo", "F003", "company")
