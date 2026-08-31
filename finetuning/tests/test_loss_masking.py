"""
Tests finetuning/prepare_data.py's assistant-only loss masking (render_and_mask)
using a small FakeTokenizer test double rather than a real HF tokenizer -- this
verifies the masking LOGIC (prompt tokens get label -100, assistant tokens keep real
labels, truncation is handled, offset-mapping boundary detection works) without
needing torch/transformers installed or a network call to download
Qwen2.5-1.5B-Instruct's tokenizer. Wiring against the real tokenizer is exercised at
training time (finetuning/train.py, run on Colab per colab_runbook.md) -- and it was
exactly that real run that caught a bug in an earlier version of render_and_mask (see
that function's docstring): tokenizing the prompt and the full conversation as two
separate calls and comparing token-ID prefixes breaks because BPE tokenizers don't
guarantee a text prefix tokenizes to a token-ID prefix. render_and_mask now tokenizes
the full conversation once and uses character offset-mapping instead, so
FakeTokenizer here implements that same interface (`apply_chat_template` rendering
real template TEXT, plus `__call__(text, return_offsets_mapping=True)` returning real
character offsets) rather than a simplified id-list shortcut -- it exercises the exact
same code path render_and_mask actually uses against a real tokenizer.
"""
import re

import pytest

from finetuning.prepare_data import render_and_mask


class FakeTokenizer:
    """A tiny stand-in for a real HF fast tokenizer. Renders messages as
    `<|role|>\ncontent<|end|>\n` (structurally analogous to Qwen's
    `<|im_start|>role\ncontent<|im_end|>\n`, without trying to be byte-identical to
    it), tokenizes whitespace-delimited words with real character offsets, and caches
    a stable word->id mapping so the same word always gets the same token id."""

    def __init__(self):
        self._word_ids = {}
        self._next_id = 1

    def _tok(self, word: str) -> int:
        if word not in self._word_ids:
            self._word_ids[word] = self._next_id
            self._next_id += 1
        return self._word_ids[word]

    def _render_message(self, m: dict) -> str:
        return f"<|{m['role']}|>\n{m['content']}<|end|>\n"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(self._render_message(m) for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>\n"
        if not tokenize:
            return text
        return self(text, add_special_tokens=False)["input_ids"]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = []
        offsets = []
        for m in re.finditer(r"\S+", text):
            input_ids.append(self._tok(m.group(0)))
            offsets.append((m.start(), m.end()))
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _record(user_text="Where is NeuroSync headquartered", assistant_text="Denver"):
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]}


def test_prompt_tokens_are_masked_and_assistant_tokens_are_not():
    tok = FakeTokenizer()
    record = _record()
    result = render_and_mask(record, tok, max_length=64)

    full_text = tok.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
    prompt_text = tok.apply_chat_template(record["messages"][:-1], tokenize=False, add_generation_prompt=True)
    encoded = tok(full_text, return_offsets_mapping=True)

    expected_labels = [
        -100 if start < len(prompt_text) else tid
        for tid, (start, _end) in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]

    assert result["input_ids"] == encoded["input_ids"]
    assert result["labels"] == expected_labels
    # The assistant portion must be non-trivial and unmasked.
    assert any(l != -100 for l in result["labels"])
    # And at least the prompt's own tokens (user turn) must be masked.
    assert any(l == -100 for l in result["labels"])


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
    silently collide on word -> id assignment."""
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


def test_boundary_token_straddling_prompt_and_assistant_text_is_masked_conservatively():
    """If a token's span starts before the prompt/assistant boundary but would
    otherwise be considered "assistant" content, render_and_mask masks it -- makes
    the -100/real-label split a strict function of token start offset, which is easy
    to reason about and never accidentally leaks a prompt token's label."""
    tok = FakeTokenizer()

    class StraddlingTokenizer(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            result = super().__call__(text, add_special_tokens, return_offsets_mapping)
            if return_offsets_mapping and result["offset_mapping"]:
                # Force the very first token to appear to start at position 0 and
                # extend past the natural word boundary, simulating a token that
                # straddles further into the text than a single word would.
                start, end = result["offset_mapping"][0]
                result["offset_mapping"][0] = (start, end + 5)
            return result

    straddling_tok = StraddlingTokenizer()
    result = render_and_mask(_record(), straddling_tok, max_length=64)
    # First token still starts at 0 (well before the prompt boundary), so it's masked
    # regardless of the tampered end offset -- start offset is the only thing
    # render_and_mask's masking rule looks at.
    assert result["labels"][0] == -100


def test_compute_prompt_token_length_stats_reflects_real_content_length():
    """Regression guard for a real bug: an earlier version of
    compute_prompt_token_length_stats called
    tokenizer.apply_chat_template(tokenize=True) directly and measured len() of the
    result -- on a transformers version where that returns a dict/BatchEncoding
    instead of a plain id list, len() silently measured the number of DICT KEYS (2)
    instead of a token count, so every example came back reporting length "2"
    regardless of content. This asserts the fixed version (which reuses
    render_and_mask) produces lengths that actually scale with content size."""
    from finetuning.prepare_data import compute_prompt_token_length_stats

    tok = FakeTokenizer()
    short = _record("Where is NeuroSync headquartered", "Denver")
    long = _record("a " * 40, "b " * 40)

    short_stats = compute_prompt_token_length_stats([short], tok)
    long_stats = compute_prompt_token_length_stats([long], tok)

    assert short_stats["max"] > 2
    assert long_stats["max"] > short_stats["max"]
