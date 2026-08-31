"""
Pure-Python test: verification/config.py's fixed generic-prompt set is small,
non-empty, and every prompt's expected_substring is a short, exact, mechanically
checkable string (never an open-ended instruction that would need an LLM judge --
locked recommendation, plan.md's Module 4 Open Decisions).
"""
from verification import config as v_config


def test_prompt_set_is_non_empty_and_bounded():
    n = len(v_config.GENERAL_CAPABILITY_PROMPTS)
    assert 5 <= n <= 20


def test_every_prompt_has_a_short_mechanically_gradable_expected_substring():
    for item in v_config.GENERAL_CAPABILITY_PROMPTS:
        assert "prompt" in item and item["prompt"].strip()
        assert "expected_substring" in item
        assert 1 <= len(item["expected_substring"]) <= 20, (
            f"expected_substring {item['expected_substring']!r} looks too long to be a "
            f"single mechanically-checkable answer -- general-capability prompts must "
            f"never need an LLM judge"
        )


def test_prompts_are_all_distinct():
    prompts = [item["prompt"] for item in v_config.GENERAL_CAPABILITY_PROMPTS]
    assert len(prompts) == len(set(prompts))
