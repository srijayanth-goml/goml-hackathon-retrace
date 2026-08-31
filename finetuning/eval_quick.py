"""
Module 2's post-training quick sanity check (plan.md step 6) -- NOT Module 4's full
verification suite. This exists purely as a fast, cheap gate that answers "did this
training run actually work" before spending more Colab time or handing off to
Module 3: forward-direction QA accuracy against data/processed/heldout.jsonl
(read-only, never trained on), overall and broken out by attribute.

Expect the baseline (revision-0) to score high across the board. Expect the
retain-only reference model to score at-or-near-chance specifically on the flagship
demo entity's own held-out QA (it never saw those facts) while matching the baseline
everywhere else -- that gap is itself informal evidence the retain-only training set
was built correctly.

Only `run_quick_eval`'s generation step needs a real transformers model+tokenizer at
call time; everything else here (loading heldout QA, looking up expected values)
needs only the repo-root requirements.txt.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import config as root_config
from data_pipeline.format_chat import read_jsonl
from data_pipeline.load import load_fact_rows


def _fact_value_map() -> Dict[str, str]:
    """fact_id -> ground-truth value, straight from the source CSV (not from the
    training text) -- so eval doesn't accidentally grade against paraphrased wording."""
    rows = load_fact_rows(root_config.RAW_CSV_PATH)
    return {r.fact_id: r.value for r in rows}


def load_forward_qa_heldout() -> List[dict]:
    records = list(read_jsonl(root_config.HELDOUT_JSONL_PATH))
    return [
        r for r in records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "forward"
    ]


def generate_answer(model, tokenizer, user_message: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False: `prompt` already contains the chat template's own
    # special tokens as literal text (<|im_start|> etc.) -- letting the tokenizer add
    # ITS OWN special tokens on top (e.g. a BOS it thinks is missing) would corrupt
    # the exact prompt boundary the model was trained against. Same reasoning as
    # prepare_data.render_and_mask.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_quick_eval(model, tokenizer, limit: "int | None" = None) -> Tuple[dict, List[dict]]:
    """Returns (summary, per_example_details). summary has overall_accuracy,
    accuracy_by_attribute, and n; per_example_details records question/expected/got/
    correct for every held-out forward-QA example, for manual spot-checking.
    `limit` caps how many held-out examples are evaluated (useful for a quick local
    CPU run)."""
    value_map = _fact_value_map()
    records = load_forward_qa_heldout()
    if limit is not None:
        records = records[:limit]

    correct_by_attr: Counter = Counter()
    total_by_attr: Counter = Counter()
    details: List[dict] = []

    for r in records:
        fact_id = r["metadata"]["fact_ids"][0]
        attribute = r["metadata"]["attribute"]
        expected_value = value_map[fact_id]
        question = r["messages"][0]["content"]

        answer = generate_answer(model, tokenizer, question)
        is_correct = expected_value.strip().lower() in answer.strip().lower()

        total_by_attr[attribute] += 1
        if is_correct:
            correct_by_attr[attribute] += 1

        details.append({
            "fact_id": fact_id,
            "entity": r["metadata"]["entity"],
            "attribute": attribute,
            "question": question,
            "expected_value": expected_value,
            "generated_answer": answer,
            "correct": is_correct,
        })

    accuracy_by_attribute = {
        attr: correct_by_attr[attr] / total_by_attr[attr] for attr in sorted(total_by_attr)
    }
    n = sum(total_by_attr.values())
    overall_accuracy = (sum(correct_by_attr.values()) / n) if n else 0.0

    summary = {
        "n": n,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_attribute": accuracy_by_attribute,
    }
    return summary, details
# --------------------------------------------------------------------------------
# CLI: run standalone, e.g. `python -m finetuning.eval_quick --which baseline`.
#
# NOTE: run this with `-m` from the repo root, not `python finetuning/eval_quick.py`.
# The latter puts finetuning/'s own directory first on sys.path instead of the repo
# root, which breaks the repo-root-relative imports this whole codebase relies on
# (`import config`, `from common.schema import ...`, etc.) -- that's exactly what
# produced the confusing AttributeError this CLI's addition was prompted by.
# --------------------------------------------------------------------------------

def _load_model_and_tokenizer(adapter_dir):
    """Loads the frozen base model + a LoRA adapter from `adapter_dir` (or the base
    model alone if adapter_dir is None), in eval mode. Mirrors finetuning.train's
    _train_one/_run_quick_eval_on setup, including the pad-token fallback that
    finetuning.train._run_quick_eval_on was previously missing (train-time and
    eval-time tokenizer setup should be identical -- a None pad_token_id reaching
    model.generate() can silently produce garbage output)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetuning import ft_config

    tokenizer = AutoTokenizer.from_pretrained(ft_config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ft_config.MODEL_NAME, torch_dtype=torch.bfloat16 if ft_config.BF16 else torch.float32
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def _resolve_adapter_dir(args) -> "Path":
    from pathlib import Path as _Path

    from finetuning import ft_config

    if args.adapter_dir:
        return _Path(args.adapter_dir)
    if args.which == "baseline":
        return ft_config.BASELINE_CHECKPOINT_DIR
    entity = args.entity or ft_config.FLAGSHIP_DEMO_ENTITY
    slug = entity.lower().replace(" ", "-")
    return ft_config.reference_checkpoint_dir(slug)


def debug_inspect_samples(model, tokenizer, base_model, n: int) -> None:
    """Prints, for `n` examples drawn from data/processed/train.jsonl (NOT
    heldout -- we want examples the model was actually trained on, so we can see
    render_and_mask's masking directly): the rendered prompt/assistant text, which
    tokens render_and_mask kept real labels on (decoded back to text, so the
    prompt/assistant boundary is visually obvious), and the generated answer from
    both the adapter model and the plain base model (no adapter) side by side.

    Use this to distinguish between two very different failure modes if quick-eval
    accuracy is unexpectedly low:
      1. Adapter output == base-model output (or nonsense in the same way) -> the
         adapter likely isn't being applied/loaded correctly, or training barely
         changed the model at all (check the label-decode output below for whether
         real training signal was being fed in the first place).
      2. Adapter output differs from base-model output but is still wrong -> the
         adapter DID learn something, so look at exact-match string comparison
         (formatting mismatch?) or generation settings before suspecting training data.
    """
    from finetuning.prepare_data import load_train_records, render_and_mask

    records = load_train_records()
    qa_forward = [
        r for r in records
        if r["metadata"]["source_type"] == "qa" and r["metadata"]["direction"] == "forward"
    ][:n]

    for i, r in enumerate(qa_forward):
        question = r["messages"][0]["content"]
        expected = r["messages"][1]["content"]

        rendered = render_and_mask(r, tokenizer, max_length=1_000_000)
        label_ids = [t for t in rendered["labels"] if t != -100]
        unmasked_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        n_prompt_tokens = sum(1 for t in rendered["labels"] if t == -100)
        n_total_tokens = len(rendered["input_ids"])

        adapter_answer = generate_answer(model, tokenizer, question)
        base_answer = generate_answer(base_model, tokenizer, question) if base_model is not None else None

        print(f"--- sample {i} ---")
        print(f"question:                {question}")
        print(f"expected (from CSV/text): {expected}")
        print(f"tokens: {n_total_tokens} total, {n_prompt_tokens} masked (prompt), "
              f"{n_total_tokens - n_prompt_tokens} real labels (assistant)")
        print(f"decoded UNMASKED labels (what the model was actually trained to predict):")
        print(f"    {unmasked_text!r}")
        print(f"adapter model generated:  {adapter_answer!r}")
        if base_answer is not None:
            print(f"base model (no adapter):  {base_answer!r}")
        print()


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["baseline", "reference"], default="baseline",
                         help="Which checkpoint to evaluate (ignored if --adapter-dir is given)")
    parser.add_argument("--adapter-dir", default=None, help="Explicit adapter directory (overrides --which)")
    parser.add_argument("--entity", default=None, help="Entity name for --which reference (default: config's flagship entity)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of held-out QA examples evaluated (useful for a quick local CPU run)")
    parser.add_argument("--debug-samples", type=int, default=0,
                         help="Also print rich per-example diagnostics (prompt/mask/generation, adapter vs. base model) for this many TRAINING examples")
    parser.add_argument("--json-out", default=None, help="Optional path to write the summary+details as JSON")
    args = parser.parse_args()

    adapter_dir = _resolve_adapter_dir(args)
    print(f"Loading base model + adapter from: {adapter_dir}")
    model, tokenizer = _load_model_and_tokenizer(adapter_dir)

    if args.debug_samples > 0:
        print(f"\n=== Loading plain base model (no adapter) for comparison ===")
        base_model, _ = _load_model_and_tokenizer(None)
        print(f"\n=== Debug samples ({args.debug_samples}) ===\n")
        debug_inspect_samples(model, tokenizer, base_model, args.debug_samples)

    print(f"\n=== Running quick eval ===")
    summary, details = run_quick_eval(model, tokenizer, limit=args.limit)

    print(json.dumps(summary, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "details": details}, indent=2))
        print(f"Wrote full details to {args.json_out}")


if __name__ == "__main__":
    from pathlib import Path
    main()
