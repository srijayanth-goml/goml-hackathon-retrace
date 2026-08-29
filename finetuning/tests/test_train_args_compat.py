"""
Regression test for a bug hit on a real Colab run: `finetuning/train.py` called
`transformers.TrainingArguments(..., warmup_ratio=..., ...)` directly, and a
transformers release newer than this code was written against (finetuning's
requirements.txt pins `transformers>=4.44` with no upper bound -- Colab always
installs whatever is newest) had dropped/renamed `warmup_ratio`, raising
`TypeError: TrainingArguments.__init__() got an unexpected keyword argument
'warmup_ratio'` and killing the whole run.

`finetuning.train._build_training_arguments` now introspects the installed class's
actual constructor signature and filters unsupported kwargs instead of crashing. This
test exercises that filtering logic directly against fake stand-in classes -- no real
transformers needed -- so a regression here is caught by `pytest` rather than only by
the next Colab run.
"""
from pathlib import Path

from finetuning.train import _build_training_arguments

_UNUSED_PATH = Path("/tmp/finetuning-test-args-compat-unused")


class _FakeArgsMissingWarmupRatio:
    """Mimics a transformers.TrainingArguments whose constructor no longer accepts
    warmup_ratio -- the exact shape of bug that broke a real training run."""

    def __init__(
        self,
        *,
        output_dir,
        num_train_epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        lr_scheduler_type,
        bf16,
        logging_steps,
        eval_strategy,
        eval_steps,
        save_strategy,
        report_to,
        seed,
        run_name,
    ):
        self.kwargs = dict(
            output_dir=output_dir, num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type,
            bf16=bf16, logging_steps=logging_steps, eval_strategy=eval_strategy,
            eval_steps=eval_steps, save_strategy=save_strategy, report_to=report_to,
            seed=seed, run_name=run_name,
        )


class _FakeArgsAcceptsEverything:
    """A stand-in whose signature explicitly names every kwarg
    _build_training_arguments might want to pass (including warmup_ratio) --
    confirms the happy path still passes them all through unchanged when the
    installed version DOES support them."""

    def __init__(
        self,
        *,
        output_dir,
        num_train_epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        lr_scheduler_type,
        warmup_ratio,
        bf16,
        logging_steps,
        eval_strategy,
        eval_steps,
        save_strategy,
        report_to,
        seed,
        run_name,
    ):
        self.kwargs = dict(
            output_dir=output_dir, num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio, bf16=bf16, logging_steps=logging_steps,
            eval_strategy=eval_strategy, eval_steps=eval_steps,
            save_strategy=save_strategy, report_to=report_to, seed=seed,
            run_name=run_name,
        )


def test_build_training_arguments_drops_unsupported_warmup_ratio_instead_of_crashing(capsys):
    args = _build_training_arguments(
        _UNUSED_PATH, "test-run", has_eval=True, training_arguments_cls=_FakeArgsMissingWarmupRatio
    )
    assert "warmup_ratio" not in args.kwargs
    assert args.kwargs["num_train_epochs"] is not None  # everything else still got through

    captured = capsys.readouterr()
    assert "warmup_ratio" in captured.out  # the drop was printed, not silent


def test_build_training_arguments_keeps_warmup_ratio_when_the_class_accepts_it():
    args = _build_training_arguments(
        _UNUSED_PATH, "test-run", has_eval=True, training_arguments_cls=_FakeArgsAcceptsEverything
    )
    assert "warmup_ratio" in args.kwargs


def test_build_training_arguments_toggles_eval_strategy_on_has_eval():
    no_eval = _build_training_arguments(
        _UNUSED_PATH, "test-run", has_eval=False, training_arguments_cls=_FakeArgsAcceptsEverything
    )
    with_eval = _build_training_arguments(
        _UNUSED_PATH, "test-run", has_eval=True, training_arguments_cls=_FakeArgsAcceptsEverything
    )
    assert no_eval.kwargs["eval_strategy"] == "no"
    assert no_eval.kwargs["eval_steps"] is None
    assert with_eval.kwargs["eval_strategy"] == "steps"
    assert with_eval.kwargs["eval_steps"] is not None
