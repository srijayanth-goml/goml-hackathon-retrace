import pytest

from unlearning.request import ErasureRequest


def test_entity_only_classifies_as_entity():
    r = ErasureRequest(entity="NeuroSync Diagnostics")
    assert r.request_type == "entity"


def test_entity_and_attribute_classifies_as_attribute_cell():
    r = ErasureRequest(entity="NeuroSync Diagnostics", attribute="ceo")
    assert r.request_type == "attribute_cell"


def test_attribute_only_classifies_as_attribute_type():
    r = ErasureRequest(attribute="ceo")
    assert r.request_type == "attribute_type"


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        ErasureRequest()


def test_to_dict_from_dict_roundtrip():
    r = ErasureRequest(entity="NeuroSync Diagnostics", attribute="ceo")
    assert ErasureRequest.from_dict(r.to_dict()) == r


def test_from_json_file(tmp_path):
    path = tmp_path / "request.json"
    path.write_text('{"entity": "NeuroSync Diagnostics"}')
    r = ErasureRequest.from_json_file(path)
    assert r == ErasureRequest(entity="NeuroSync Diagnostics")


def test_example_request_files_all_load():
    import json

    from unlearning import config as ul_config
    for path in sorted(ul_config.REQUESTS_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        if raw.get("_deprecated"):
            continue  # e.g. silvergate_aerospace_entity.json -- see its own _comment
        r = ErasureRequest.from_json_file(path)
        assert r.request_type in ("entity", "attribute_cell", "attribute_type")


def test_deprecated_example_request_is_flagged_not_silently_valid():
    """silvergate_aerospace_entity.json is kept as a documented dead end (Silvergate
    Aerospace is a Module 1 heldout entity -- see its _comment) rather than deleted;
    confirm it's actually marked, not just assumed to be."""
    from unlearning import config as ul_config
    import json

    path = ul_config.REQUESTS_DIR / "silvergate_aerospace_entity.json"
    raw = json.loads(path.read_text())
    assert raw.get("_deprecated") is True
