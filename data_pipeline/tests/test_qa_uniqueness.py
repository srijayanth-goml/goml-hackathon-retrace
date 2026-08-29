import config
from data_pipeline.augment.qa import build_qa_examples
from data_pipeline.load import load_fact_rows


def test_isabel_ortiz_ceo_duplicate_has_no_reverse_example():
    """Review doc: 'Isabel Ortiz' is CEO of both Solara Grid and Helion Power. A
    reverse-QA example ('Which company has Isabel Ortiz as CEO?') would have exactly
    two correct answers, so the skip policy must exclude both, not emit either as a
    single-answer example."""
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    examples, stats = build_qa_examples(fact_rows)

    reverse_ceo_answers = [
        ex.messages[1]["content"]
        for ex in examples
        if ex.metadata.source_type == "qa"
        and ex.metadata.direction == "reverse"
        and ex.metadata.attribute == "ceo"
    ]
    for answer in reverse_ceo_answers:
        assert "Isabel Ortiz" not in answer

    assert stats["reverse_qa_skipped_non_unique"].get("ceo", 0) >= 2


def test_forward_qa_covers_every_fact_id():
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    examples, _ = build_qa_examples(fact_rows)
    forward_fact_ids = {
        ex.metadata.fact_ids[0]
        for ex in examples
        if ex.metadata.source_type == "qa" and ex.metadata.direction == "forward"
    }
    assert forward_fact_ids == {r.fact_id for r in fact_rows}


def test_some_reverse_qa_examples_are_still_generated():
    """The skip policy should only skip non-unique values -- most attribute values
    in this dataset are unique to one entity, so reverse QA shouldn't disappear
    entirely."""
    fact_rows = load_fact_rows(config.RAW_CSV_PATH)
    examples, _ = build_qa_examples(fact_rows)
    reverse_examples = [
        ex for ex in examples if ex.metadata.source_type == "qa" and ex.metadata.direction == "reverse"
    ]
    assert len(reverse_examples) > 0
