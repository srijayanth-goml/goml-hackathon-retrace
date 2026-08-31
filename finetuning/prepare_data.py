"""
Module 2 data preparation: loads data/processed/train.jsonl (Module 1's output) and
produces everything finetuning/train.py needs:

  1. A train/validation split for training-time loss monitoring -- Module 2's OWN
     split, separate from Module 1's train/heldout split, since data_pipeline/split.py
     does not produce one (see plan.md's "Two gaps Module 1 left for this module to
     close"). Reuses data_pipeline.split's exact stratified-by-fact_group_id method
     rather than reimplementing it, so this doesn't silently diverge from Module 1's
     tested logic.
  2. The retain-only filtered training set used to train the reference model: every
     example is dropped if it mentions the flagship demo entity's fact_group_id at all
     -- as its own subject OR as a mentioned_entities reference in a relational example.
     This is deliberately stricter than Module 3's later redact-don't-drop policy for
     relational examples during unlearning (see step 3's docstring below for why).
  3. Chat-template rendering + assistant-only loss masking, generic over any tokenizer
     that implements HF's `apply_chat_template(messages, tokenize=..., add_generation_prompt=...)`
     interface -- this is the manual-masking fallback Design Doc Section 5 anticipates
     needing if TRL's built-in completion-only-loss support isn't used instead.

Nothing in this module requires torch/transformers/peft at import time -- only
`render_and_mask` and `compute_prompt_token_length_stats` take a tokenizer object as
an argument when called, so this file (and its tests) import cleanly with just the
repo-root requirements.txt installed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import config as root_config
from common.schema import ChatExample, ExampleMetadata
from data_pipeline.format_chat import read_jsonl
from data_pipeline.split import assign_example_splits, assign_group_splits
from finetuning import ft_config

Record = dict  # a {"messages": [...], "metadata": {...}} dict, as read from train.jsonl


def load_train_records() -> List[Record]:
    """All of Module 1's train.jsonl, as plain dicts (not yet filtered or split)."""
    return list(read_jsonl(root_config.TRAIN_JSONL_PATH))


# --------------------------------------------------------------------------------
# 1. Train/validation split for training-time monitoring (Module 2's own; see module
#    docstring). Reuses data_pipeline.split.assign_group_splits/assign_example_splits
#    directly by round-tripping records through the ChatExample/ExampleMetadata
#    dataclasses Module 1 already defined -- not a parallel reimplementation.
# --------------------------------------------------------------------------------

def _record_to_chat_example(record: Record) -> ChatExample:
    md = record["metadata"]
    return ChatExample(
        messages=record["messages"],
        metadata=ExampleMetadata(
            fact_ids=md["fact_ids"],
            fact_group_ids=md["fact_group_ids"],
            source_type=md["source_type"],
            split=md.get("split"),
            entity=md.get("entity"),
            entity_type=md.get("entity_type"),
            attribute=md.get("attribute"),
            direction=md.get("direction"),
            template=md.get("template"),
            mentioned_entities=md.get("mentioned_entities"),
            cluster_axis=md.get("cluster_axis"),
        ),
    )


def _all_fact_group_ids_by_type() -> Dict[str, List[str]]:
    """Every fact_group_id in the dataset (all 100, train AND heldout), keyed by
    entity_type, read straight from the source CSV -- not just the ones with a
    single-entity example in train.jsonl. This matters because a relational example
    that survived Module 1's train/heldout policy (train unless every mentioned group
    is heldout) can still reference a group that IS in Module 1's heldout split and
    therefore has no single-entity example of its own anywhere in train.jsonl;
    assign_example_splits (below) needs a split label for every fact_group_id any
    record's metadata.fact_group_ids mentions, not just the ones train.jsonl "owns"."""
    from data_pipeline.load import group_by_fact_group, load_fact_rows

    rows = load_fact_rows(root_config.RAW_CSV_PATH)
    groups = group_by_fact_group(rows)
    by_type: Dict[str, List[str]] = defaultdict(list)
    for gid, group_rows in groups.items():
        by_type[group_rows[0].entity_type].append(gid)
    return {t: sorted(gids) for t, gids in by_type.items()}


def split_records_for_sft(
    records: Sequence[Record],
    val_fraction: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Record], List[Record]]:
    """Returns (sft_train_records, sft_val_records). Splits BY fact_group_id, stratified
    by entity_type, using the same method as Module 1's assign_group_splits -- a
    relational example spanning multiple groups goes to sft_val only if every group it
    mentions is in sft_val (identical policy to Module 1's train/heldout split, applied
    here to a different split for a different purpose)."""
    val_fraction = ft_config.TRAIN_VAL_FRACTION if val_fraction is None else val_fraction
    seed = ft_config.TRAIN_VAL_SEED if seed is None else seed

    examples = [_record_to_chat_example(r) for r in records]
    fact_group_ids_by_type = _all_fact_group_ids_by_type()

    # Reuse assign_group_splits/assign_example_splits with THEIR OWN "train"/"heldout"
    # vocabulary (assign_example_splits hardcodes the string "heldout" in its
    # all-mentioned-groups-must-be-heldout check for relational examples -- relabeling
    # to "sft_train"/"val" before calling it would silently break that check and drop
    # every relational example from both buckets). We only reinterpret the labels
    # ("heldout" -> our validation slice, "train" -> our sft-training slice) when
    # bucketing records afterward, below.
    split_of_group = assign_group_splits(fact_group_ids_by_type, val_fraction, seed)
    assign_example_splits(examples, split_of_group)  # mutates each ex.metadata.split in place

    train_records = [rec for rec, ex in zip(records, examples) if ex.metadata.split == "train"]
    val_records = [rec for rec, ex in zip(records, examples) if ex.metadata.split == "heldout"]
    return train_records, val_records


# --------------------------------------------------------------------------------
# 2. Retain-only filter for the reference model.
# --------------------------------------------------------------------------------

def build_retain_only_records(
    records: Sequence[Record], excluded_fact_group_id: str
) -> Tuple[List[Record], Dict[str, int]]:
    """Drops every record that mentions `excluded_fact_group_id` at all -- as its own
    subject (metadata.fact_group_ids) or as a relational co-mention
    (metadata.mentioned_entities) -- and returns (kept_records, drop_counts_by_source_type).

    This is intentionally stricter than Module 3's later redact-don't-drop policy for
    relational examples that mention a forgotten entity alongside a retained sibling
    (CLAUDE.md's open relational-example question): that policy governs what to do with
    retain data while erasing facts from an ALREADY-TRAINED model. This function instead
    builds a model from scratch that must be able to claim it never saw the target
    entity's facts at all -- redacting a mention (leaving a same-shaped sentence with
    the name blanked) does not support that claim as cleanly as dropping the example
    entirely, so dropping is the only option used here.

    Does NOT touch the excluded entity's confusable neighbors (e.g. NeuroWave,
    NeuroCore stay fully in the retain-only set) -- only the named entity's own facts
    and any sentence that names it are removed.
    """
    kept: List[Record] = []
    dropped: List[Record] = []
    for r in records:
        md = r["metadata"]
        fact_group_ids = set(md.get("fact_group_ids") or [])
        mentioned = set(md.get("mentioned_entities") or [])
        if excluded_fact_group_id in fact_group_ids or excluded_fact_group_id in mentioned:
            dropped.append(r)
        else:
            kept.append(r)

    drop_counts = Counter(r["metadata"]["source_type"] for r in dropped)
    return kept, dict(drop_counts)


# --------------------------------------------------------------------------------
# 3. Chat-template rendering + assistant-only loss masking.
# --------------------------------------------------------------------------------

def render_and_mask(record: Record, tokenizer, max_length: int) -> Dict[str, List[int]]:
    """Renders one {"messages": [user, assistant]} record through `tokenizer`'s chat
    template and masks every token through the end of the prompt (system + user turns,
    plus the generation-prompt marker) with label -100, keeping real labels only on the
    assistant's response tokens -- Design Doc Section 5's assistant-only loss masking.

    IMPORTANT: this does NOT tokenize the prompt and the full conversation as two
    separate calls and compare token-ID prefixes -- an earlier version of this function
    did exactly that, and it broke on a real run (Qwen2.5-1.5B-Instruct on Colab): BPE
    tokenizers are not guaranteed to tokenize a prefix of a string the same way whether
    or not more text follows it (a token near the boundary can merge differently), so
    `tokenize(prompt)` is not reliably a prefix of `tokenize(prompt + completion)` at
    the TOKEN level even though it obviously is at the TEXT level. The fix: tokenize
    the full conversation exactly ONCE, and use the tokenizer's character
    offset-mapping (`return_offsets_mapping=True`, needs a fast/Rust tokenizer -- true
    for Qwen2.5's) to find which tokens fall before vs. after the prompt/assistant
    text boundary, rather than re-tokenizing the prompt separately and hoping the IDs
    line up.

    `tokenizer` needs `apply_chat_template(messages, tokenize=bool,
    add_generation_prompt=bool)` (returning a string when tokenize=False) plus
    `__call__(text, add_special_tokens=False, return_offsets_mapping=True)` returning a
    dict with `input_ids` and `offset_mapping` -- satisfied by a real HF fast
    tokenizer, and by the FakeTokenizer test double in
    finetuning/tests/test_loss_masking.py.
    """
    messages = record["messages"]
    if len(messages) < 2:
        raise ValueError(f"expected at least a user+assistant turn, got {messages!r}")

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    if not full_text.startswith(prompt_text):
        raise ValueError(
            "the prompt-only chat-template rendering is not a text prefix of the "
            "full-conversation rendering -- cannot locate the assistant-only loss "
            "mask boundary. This would mean the chat template does something "
            "content-dependent beyond straightforward concatenation of turns "
            "(unexpected for Qwen2.5-Instruct's template; investigate before training)."
        )
    prompt_char_len = len(prompt_text)

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])

    labels = list(input_ids)
    for i, (start, _end) in enumerate(offsets):
        # Mask any token whose span starts before the assistant's response begins.
        # A token straddling the boundary (rare -- would need a BPE merge across a
        # newline/special-token, which Qwen2.5's template avoids by construction) is
        # conservatively masked too, at the cost of at most one real label token.
        if start < prompt_char_len:
            labels[i] = -100

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def compute_prompt_token_length_stats(records: Sequence[Record], tokenizer) -> Dict[str, int]:
    """Full-conversation token-length percentiles, used to sanity-check
    finetuning/ft_config.py's MAX_SEQ_LENGTH against the actual data (step 5 of the plan:
    "measure the actual p99 token length ... and raise this if it's not comfortably
    covered").

    Reuses render_and_mask (with an effectively unbounded max_length, so nothing gets
    truncated) rather than calling `tokenizer.apply_chat_template(..., tokenize=True,
    ...)` directly and measuring `len()` of the result. An earlier version did exactly
    that, and it silently broke on a real Colab run: some transformers versions have
    `apply_chat_template(tokenize=True)` return a dict/BatchEncoding rather than a
    plain list of token ids, so `len(...)` was measuring the number of DICT KEYS (2:
    input_ids + attention_mask) instead of a token count -- every example's reported
    length came back as a suspicious flat "2", which the resulting report should have
    made obvious but didn't get caught before that training run. Going through
    render_and_mask sidesteps the whole ambiguity: it never calls apply_chat_template
    with tokenize=True (see its docstring), so there's nothing here to get confused by.
    """
    lengths = sorted(
        len(render_and_mask(r, tokenizer, max_length=1_000_000)["input_ids"])
        for r in records
    )
    n = len(lengths)
    if n == 0:
        return {"n": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}

    def pct(p: float) -> int:
        idx = min(n - 1, int(p * n))
        return lengths[idx]

    return {"n": n, "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99), "max": lengths[-1]}
