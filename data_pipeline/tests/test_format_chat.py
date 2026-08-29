import tempfile
from pathlib import Path

from common.schema import ChatExample, ExampleMetadata
from data_pipeline.format_chat import read_jsonl, write_jsonl


def test_round_trip_preserves_messages_and_metadata():
    examples = [
        ChatExample(
            messages=[
                {"role": "user", "content": "Where is NeuroSync Diagnostics headquartered?"},
                {"role": "assistant", "content": "NeuroSync Diagnostics is headquartered in Denver."},
            ],
            metadata=ExampleMetadata(
                fact_ids=["F002"],
                fact_group_ids=["G001"],
                source_type="qa",
                split="train",
                entity="NeuroSync Diagnostics",
                entity_type="company",
                attribute="headquarters",
                direction="forward",
            ),
        )
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        write_jsonl(examples, path)
        records = list(read_jsonl(path))

    assert len(records) == 1
    record = records[0]
    assert record["messages"] == examples[0].messages
    assert record["metadata"]["fact_ids"] == ["F002"]
    assert record["metadata"]["split"] == "train"
    assert record["metadata"]["direction"] == "forward"
    # Fields not set on this example should still be present as null, not missing.
    assert record["metadata"]["cluster_axis"] is None
    assert record["metadata"]["mentioned_entities"] is None


def test_matches_design_doc_record_shape():
    """Design Doc Section 5's example record is {"messages": [...], "metadata": {...}}
    -- confirm ChatExample.to_record() produces exactly that top-level shape."""
    ex = ChatExample(
        messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        metadata=ExampleMetadata(fact_ids=["F001"], fact_group_ids=["G001"], source_type="paraphrase"),
    )
    record = ex.to_record()
    assert set(record.keys()) == {"messages", "metadata"}
    assert isinstance(record["messages"], list)
    assert isinstance(record["metadata"], dict)
