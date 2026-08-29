"""
Tests finetuning/prepare_data.py's assistant-only loss masking (render_and_mask)
using a small FakeTokenizer test double rather than a real HF tokenizer -- this
verifies the masking LOGIC (prompt tokens get label -100, assistant tokens keep real
labels, truncation is handled) without needing torch/transformers installed or a
network call to download Qwen2.5-1.5B-Instruct's tokenizer. Wiring against the real
tokenizer is exercised at training time (finetuning/train.py, run on Colab per
colab_runbook.md), not by this fast unit test.

FakeTokenizer's apply_chat_template is deterministic and per-message: each message's
tokens depend only on its own (role, content), never on what else is in the list, and
a generation-prompt marker is appended only when requested. That's the property
render_and_mask actually relies on (tokenizing messages[:-1] with
add_generation_prompt=True must be an exact prefix of tokenizing the full messages
list) -- a real chat template has the same property for any template that doesn't
reorder or summarize earlier turns, which Qwen2.5-Instruct's doesn't.
"""
import pytest

from finetuning.prepare_data import render_and_mask

ROLE_MARKERS = {"system": -3, "user": -1, "assistant": -2}


class FakeTokenizer:
    def __init__(self):
        self._word_ids = {}
        self._next_id = 1

    def _tokenize_word(self, word: str) -> int:
        if word not in self._word_ids:
            self._word_ids[word] = self._next_id
            self._next_id += 1
        return self._word_ids[word]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for m in messages:
            ids.append(ROLE_MARKERS[m["role"]])
            ids.extend(self._tokenize_word(w) for w in m["content"].split())
        if add_generation_prompt:
            ids.append(ROLE_MARKERS["assistant"])
        if tokenize:
            return ids
        return " ".join(str(i) for i in ids)


def _record(user_text="Where is NeuroSync headquartered", assistant_text="Denver"):
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]}


def test_prompt_tokens_are_masked_and_assistant_tokens_are_not():
    tok = FakeTokenizer()
    record = _record()
    result = render_and_mask(record, tok, max_length=64)

    full_ids = tok.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=False)
    prompt_ids = tok.apply_chat_template(record["messages"][:-1], tokenize=True, add_generation_prompt=True)

    assert result["input_ids"] == full_ids
    assert result["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert result["labels"][len(prompt_ids):] == full_ids[len(prompt_ids):]
    # The assistant portion must be non-trivial and unmasked.
    assert any(l != -100 for l in result["labels"])


def test_attention_mask_is_all_ones_and_matches_input_length():
    tok = FakeTokenizer()
    result = render_and_mask(_record(), tok, max_length=64)
    assert result["attention_mask"] == [1] * len(result["input_ids"])


def test_truncation_keeps_input_ids_and_labels_aligned():
    tok = FakeTokenizer()
    record = _record(user_text="a " * 50, assistant_text="b " * 50)
    max_length = 10
    result = render_and_mask(record, tok, max_length=max_length)
    assert len(result["input_ids"]) == max_length
    assert len(result["labels"]) == max_length
    assert len(result["attention_mask"]) == max_length


def test_different_examples_do_not_bleed_into_each_others_token_ids():
    """Regression guard for FakeTokenizer itself: two different records must not
    silently collide on word -> id assignment in a way that breaks the prefix
    property render_and_mask depends on."""
    tok = FakeTokenizer()
    r1 = _record("Where is NeuroSync headquartered", "Denver")
    r2 = _record("Who is the CEO of NeuroWave", "Alex Rivera")
    result1 = render_and_mask(r1, tok, max_length=64)
    result2 = render_and_mask(r2, tok, max_length=64)
    assert result1["input_ids"] != result2["input_ids"]


def test_rejects_a_record_with_only_one_message():
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        render_and_mask({"messages": [{"role": "user", "content": "hi"}]}, tok, max_length=64)
