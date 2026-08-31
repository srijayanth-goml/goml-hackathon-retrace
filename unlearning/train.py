"""
Module 3 (Unlearning Scripts) entrypoint (Design Doc Section 6 / plan.md's detailed
plan). Runs LOCALLY against the Colab-trained baseline adapter -- NOT on Colab, per
../CLAUDE.md's architecture decision.

    python -m unlearning.train --request unlearning/requests/neurosync_entity.json --method npo
    python -m unlearning.train --entity "NeuroSync Diagnostics" --method ga

One training loop, parameterized by --method (npo/ga), so a judge comparing the two
adapters produced for the same request is comparing "which forgetting method", not
"different code paths" -- the same posture Module 2's baseline/reference split
already established for ITS comparison.

Needs torch/transformers/peft installed (see the repo-root requirements.txt) -- NOT
necessarily a GPU: a 1.5B model in bf16 LoRA runs (slowly) on CPU or Apple
Silicon/MPS too, which matters since this module is specified to run locally
(Design Doc Section 10's open hardware question).

Model loading (loading pi_ref + pi_theta from the same adapter) and batch collation
are implemented in unlearning/model_io.py, not here -- extracted so
verification/*.py (Module 4) can load a single adapter for eval without duplicating
this logic (plan.md's Module 4 Open Decisions: "a pure extraction, no behavior
change to Module 3's training path").
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import config as root_config
from finetuning import ft_config
from unlearning import config as ul_config
from unlearning import eval_during_unlearning as ev
from unlearning import manifest as ul_manifest
from unlearning import model_io
from unlearning.data import build_unlearning_batches, forget_sampler, neighbor_weighted_sampler
from unlearning.request import ErasureRequest

REPO_ROOT = ul_config.UNLEARNING_DIR.parent


def _require_heavy_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"unlearning/train.py needs {missing} installed -- see the repo-root "
            f"requirements.txt (now includes torch/transformers/peft/accelerate for "
            f"exactly this module -- see plan.md's Module 3 decisions) and run "
            f"`pip install -r requirements.txt`. This is expected to fail before "
            f"those are installed; it does NOT require a GPU (a 1.5B model in bf16 "
            f"LoRA runs, slowly, on CPU or Apple Silicon/MPS too)."
        )


def _relative_path(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT))


def _sample_general(records: List[dict], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    if len(records) <= n:
        return list(records)
    return rng.sample(records, n)


def _pre_unlearning_baseline(theta_model, tokenizer, batches) -> dict:
    """Accuracy snapshot BEFORE any unlearning step -- what the early-stop-on-
    neighbor-drift rule compares every later snapshot against (Design Doc Section 6
    step 4: neighbor/general accuracy targets are "flat", meaning flat relative to
    THIS, not to some absolute number)."""
    theta_model.eval()
    general_sample = _sample_general(batches.retain_general, n=ul_config.GENERAL_EVAL_SAMPLE_SIZE, seed=ul_config.SEED)
    result = ev.track_all(theta_model, tokenizer, batches.forget_train, batches.retain_neighbor, general_sample, batches.forget_probe)
    theta_model.train()
    return result


def _should_early_stop(baseline: dict, current: dict) -> bool:
    forget_acc = current["forget"]["overall_accuracy"]
    if forget_acc is None or forget_acc > ul_config.FORGET_ACCURACY_COLLAPSE_THRESHOLD:
        return False

    def _drifted(pool_name: str, tolerance: float) -> bool:
        before = baseline[pool_name]["overall_accuracy"]
        after = current[pool_name]["overall_accuracy"]
        if before is None or after is None:
            return False
        return (before - after) > tolerance

    if _drifted("neighbor", ul_config.NEIGHBOR_DRIFT_TOLERANCE):
        return False
    if _drifted("general", ul_config.GENERAL_DRIFT_TOLERANCE):
        return False
    return True


def _train_step(method, theta_model, ref_model, tokenizer, forget_records, retain_records, beta, lambda_retain):
    from unlearning.npo import compute_batch_logps, npo_loss_tensor
    import torch

    forget_batch = model_io.pad_collate(forget_records, tokenizer, ft_config.MAX_SEQ_LENGTH, tokenizer.pad_token_id)
    retain_batch = model_io.pad_collate(retain_records, tokenizer, ft_config.MAX_SEQ_LENGTH, tokenizer.pad_token_id)

    if method == "npo":
        theta_forget_logps = compute_batch_logps(theta_model, forget_batch)
        with torch.no_grad():
            ref_forget_logps = compute_batch_logps(ref_model, forget_batch)
        forget_loss = npo_loss_tensor(theta_forget_logps, ref_forget_logps, beta)
    elif method == "ga":
        from unlearning.gradient_ascent import ga_loss_tensor
        forget_loss = ga_loss_tensor(theta_model, forget_batch)
    else:
        raise ValueError(f"unknown method {method!r}")

    retain_logps = compute_batch_logps(theta_model, retain_batch)
    retain_loss = -retain_logps.mean()  # ordinary SFT loss = -logp

    # Design Doc Section 6: GA is run WITHOUT a retain term, on purpose -- it has no
    # brake, and part of the point of running it is to show what happens without one.
    total = forget_loss + lambda_retain * retain_loss if method == "npo" else forget_loss

    return total, float(forget_loss.detach()), float(retain_loss.detach())


def run(
    request: ErasureRequest,
    method: str,
    parent_revision: Optional[int] = None,
    max_steps: Optional[int] = None,
    skip_final_eval: bool = False,
) -> dict:
    _require_heavy_deps()
    import torch

    parent_revision = ul_config.DEFAULT_PARENT_REVISION if parent_revision is None else parent_revision
    max_steps = ul_config.MAX_STEPS if max_steps is None else max_steps

    adapter_dir = (
        ft_config.BASELINE_CHECKPOINT_DIR
        if parent_revision == 0
        else REPO_ROOT / ul_manifest.load_revision_adapter_path(parent_revision)
    )

    print(f"[unlearning:{method}] request={request.to_dict()} parent_revision={parent_revision}")
    batches = build_unlearning_batches(request)
    print(f"[unlearning:{method}] {batches.summary()}")

    theta_model, ref_model, tokenizer = model_io.load_ref_and_theta(adapter_dir, ul_config.MODEL_NAME, bf16=ft_config.BF16)

    accuracy_before = _pre_unlearning_baseline(theta_model, tokenizer, batches)
    print(f"[unlearning:{method}] pre-unlearning accuracy: {accuracy_before}")

    rng = random.Random(ul_config.SEED)
    forget_iter = forget_sampler(batches.forget_train, rng)
    retain_iter = neighbor_weighted_sampler(batches.retain_general, batches.retain_neighbor, rng)

    optimizer = torch.optim.AdamW(
        (p for p in theta_model.parameters() if p.requires_grad),
        lr=ul_config.LEARNING_RATE if method == "npo" else ul_config.GA_LEARNING_RATE,
    )

    log_history: List[dict] = []
    early_stop_step: Optional[int] = None
    accuracy_after = accuracy_before

    for step in range(1, max_steps + 1):
        forget_batch_records = [next(forget_iter) for _ in range(ul_config.FORGET_BATCH_SIZE)]
        retain_batch_records = [
            next(retain_iter)
            for _ in range(ul_config.RETAIN_GENERAL_PER_FORGET + ul_config.RETAIN_NEIGHBOR_PER_FORGET)
        ]

        total_loss, forget_loss_val, retain_loss_val = _train_step(
            method, theta_model, ref_model, tokenizer, forget_batch_records, retain_batch_records,
            ul_config.NPO_BETA, ul_config.LAMBDA_RETAIN,
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log_history.append({"step": step, "forget_loss": forget_loss_val, "retain_loss": retain_loss_val})

        if step % ul_config.EVAL_EVERY_N_STEPS == 0:
            theta_model.eval()
            general_sample = _sample_general(batches.retain_general, n=ul_config.GENERAL_EVAL_SAMPLE_SIZE, seed=ul_config.SEED + step)
            accuracy_after = ev.track_all(theta_model, tokenizer, batches.forget_train, batches.retain_neighbor, general_sample, batches.forget_probe)
            theta_model.train()
            print(f"[unlearning:{method}] step={step} accuracy={accuracy_after}")
            if _should_early_stop(accuracy_before, accuracy_after):
                early_stop_step = step
                print(f"[unlearning:{method}] early-stopping at step {step}: forget collapsed, neighbor/general flat")
                break

    if not skip_final_eval:
        theta_model.eval()
        accuracy_after = ev.track_all(
            theta_model, tokenizer, batches.forget_train, batches.retain_neighbor, batches.retain_general, batches.forget_probe
        )

    revision = ul_manifest.next_revision_number()
    output_dir = ul_config.revision_checkpoint_dir(revision, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = batches.summary()
    dataset_info = {"train_jsonl_sha256": _train_jsonl_sha256(), **{k: v for k, v in summary.items() if k not in ("request", "entity_type")}}

    manifest_entry = ul_manifest.write_revision_entry(
        revision=revision,
        parent_revision=parent_revision,
        method=method,
        erasure_request=request.to_dict(),
        adapter_path=Path(_relative_path(output_dir)),
        base_model=ul_config.MODEL_NAME,
        lora_config=_adapter_lora_config(adapter_dir),
        training_args=ul_config.training_args_as_dict(method),
        dataset_info=dataset_info,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        early_stop_step=early_stop_step,
    )

    _write_report(revision, method, manifest_entry, summary, log_history, accuracy_before, accuracy_after, early_stop_step)
    print(f"[unlearning:{method}] adapter saved to {output_dir}, registered as revision-{revision}")
    return manifest_entry


def _adapter_lora_config(adapter_dir: Path) -> dict:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _train_jsonl_sha256() -> str:
    from finetuning.manifest import sha256_of_file
    return sha256_of_file(root_config.TRAIN_JSONL_PATH)


def _write_report(revision, method, manifest_entry, summary, log_history, accuracy_before, accuracy_after, early_stop_step) -> None:
    ul_config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ul_config.REPORTS_DIR / f"revision-{revision}_{method}_report.json"
    md_path = ul_config.REPORTS_DIR / f"revision-{revision}_{method}_report.md"

    report = {
        "revision": revision,
        "method": method,
        "created_at": manifest_entry["created_at"],
        "erasure_request": manifest_entry["erasure_request"],
        "summary": summary,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "early_stop_step": early_stop_step,
        "training_args": manifest_entry["training_args"],
        "log_history": log_history,
    }
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        f"# Module 3 unlearning report -- revision-{revision} ({method})",
        "",
        f"Request: `{json.dumps(manifest_entry['erasure_request'])}`",
        f"Created: {manifest_entry['created_at']}",
        f"Early-stop step: {early_stop_step if early_stop_step is not None else 'did not trigger (ran to max_steps)'}",
        "",
        "## Forget/retain set sizes", "",
        "```json", json.dumps(summary, indent=2), "```", "",
        "## Accuracy before unlearning", "",
        "```json", json.dumps(accuracy_before, indent=2), "```", "",
        "## Accuracy after unlearning", "",
        "```json", json.dumps(accuracy_after, indent=2), "```", "",
    ]
    md_path.write_text("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default=None, help="Path to an ErasureRequest JSON file (see unlearning/requests/)")
    parser.add_argument("--entity", default=None, help="Erasure request entity filter (alternative to --request)")
    parser.add_argument("--attribute", default=None, help="Erasure request attribute filter (alternative to --request)")
    parser.add_argument("--method", choices=["npo", "ga"], required=True)
    parser.add_argument("--parent-revision", type=int, default=None, help="Defaults to unlearning/config.py's DEFAULT_PARENT_REVISION (0 -- branch fresh from the baseline)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip the full final accuracy pass (faster smoke test)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.request:
        request = ErasureRequest.from_json_file(args.request)
    elif args.entity or args.attribute:
        request = ErasureRequest(entity=args.entity, attribute=args.attribute)
    else:
        raise SystemExit("pass --request <path> or --entity/--attribute")

    run(request, method=args.method, parent_revision=args.parent_revision, max_steps=args.max_steps, skip_final_eval=args.skip_final_eval)


if __name__ == "__main__":
    main()
