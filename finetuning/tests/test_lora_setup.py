"""
Tests finetuning/lora_setup.py. The target_modules check is pure Python (no peft
needed) since it only reads finetuning/config.py's constant; building an actual
peft.LoraConfig is skipped when peft isn't installed (expected on a laptop with no
GPU/ML stack -- see ../requirements.txt and ../colab_runbook.md) rather than failing
the whole test suite.
"""
import pytest

from finetuning import config as ft_config
from finetuning.lora_setup import build_lora_config


def test_target_modules_cover_attention_and_mlp_per_claude_md():
    """CLAUDE.md: "targeting both attention projections (q/k/v/o_proj) and MLP
    projections (gate/up/down_proj)" -- attention-only would silently narrow what
    the adapter can learn (Design Doc Section 5's whole argument for including MLP)."""
    attention = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}
    target = set(ft_config.LORA_TARGET_MODULES)
    assert attention <= target
    assert mlp <= target
    assert target == attention | mlp  # nothing extra, nothing missing


def test_rank_is_within_claude_md_range():
    assert 16 <= ft_config.LORA_RANK <= 32


def test_lora_config_as_dict_matches_the_constants():
    d = ft_config.lora_config_as_dict()
    assert d["r"] == ft_config.LORA_RANK
    assert d["lora_alpha"] == ft_config.LORA_ALPHA
    assert set(d["target_modules"]) == set(ft_config.LORA_TARGET_MODULES)


def test_build_lora_config_matches_config_py_when_peft_is_installed():
    pytest.importorskip("peft")
    cfg = build_lora_config()
    assert cfg.r == ft_config.LORA_RANK
    assert cfg.lora_alpha == ft_config.LORA_ALPHA
    assert set(cfg.target_modules) == set(ft_config.LORA_TARGET_MODULES)
    assert cfg.lora_dropout == ft_config.LORA_DROPOUT
