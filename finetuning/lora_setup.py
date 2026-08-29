"""
Builds the PEFT LoraConfig from finetuning/config.py's hyperparameters. Kept as its
own thin module (rather than inlined in train.py) so both train.py and the tests can
import it, and so `finetuning/config.py`'s LORA_* constants stay the single source of
truth for what "the shared LoRA adapter" (CLAUDE.md) actually targets.

`peft` is imported lazily, inside build_lora_config(), so importing this module (and
checking finetuning.config.LORA_TARGET_MODULES directly) never requires peft to be
installed -- only actually building a real LoraConfig does. See
finetuning/tests/test_lora_setup.py.
"""
from __future__ import annotations

from finetuning import config as ft_config


def build_lora_config():
    """Returns a peft.LoraConfig built from finetuning/config.py. Requires `peft`
    installed (see finetuning/requirements.txt / colab_runbook.md)."""
    from peft import LoraConfig, TaskType

    task_type = getattr(TaskType, ft_config.LORA_TASK_TYPE)
    return LoraConfig(
        r=ft_config.LORA_RANK,
        lora_alpha=ft_config.LORA_ALPHA,
        lora_dropout=ft_config.LORA_DROPOUT,
        target_modules=list(ft_config.LORA_TARGET_MODULES),
        bias=ft_config.LORA_BIAS,
        task_type=task_type,
    )


def verify_target_modules_present(model) -> None:
    """Sanity check to run right after loading the base model on Colab, before
    wrapping it with get_peft_model: confirms every name in LORA_TARGET_MODULES
    actually matches at least one module in the loaded model. Design Doc Section 5 /
    plan.md step 1 flags this explicitly -- a typo'd target-module name silently
    trains zero adapters on the intended layer instead of raising an error, so this
    check exists to make that failure loud instead of silent."""
    module_name_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [m for m in ft_config.LORA_TARGET_MODULES if m not in module_name_suffixes]
    if missing:
        raise ValueError(
            f"LORA_TARGET_MODULES names not found in the loaded model: {missing}. "
            f"Check finetuning/config.py against the actual module names in "
            f"{ft_config.MODEL_NAME} (model.named_modules())."
        )
