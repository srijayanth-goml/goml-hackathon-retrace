#!/usr/bin/env python3
"""Local demo UI for querying and checking ReTrace SISA adapters."""

from __future__ import annotations

import os
import gc
from typing import Any, Dict, Generator, List, Tuple

import gradio as gr
import yaml

from sisa.benchmark import PROBE_TYPE_OPTIONS, evaluate_in_domain_accuracy, load_shard_records, select_probe_records
from sisa.model import ModelManager
from sisa.sharding import SISAShardManager
from sisa.unlearner import SISAUnlearner


ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "sisa_config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
    CONFIG: Dict[str, Any] = yaml.safe_load(stream) or {}

PATHS = CONFIG.get("paths", {})
MODEL_NAME = CONFIG.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = CONFIG.get("model", {}).get("max_seq_length", 512)
SHARDS_DIR = os.path.join(ROOT, PATHS.get("shards_dir", "outputs/shards"))
CHECKPOINTS_DIR = os.path.join(ROOT, PATHS.get("checkpoints_dir", "outputs/checkpoints"))
UNLEARNED_CHECKPOINTS_DIR = os.path.join(ROOT, PATHS.get("unlearned_checkpoints_dir", "outputs/checkpoints_unlearned"))
SHARD_METADATA = SISAShardManager.load_metadata(SHARDS_DIR)
GROUP_LOCATIONS = SHARD_METADATA["group_locations"]
GROUP_CHOICES = [
    (f"{group_id} - {location['entity']} (Shard {location['shard_id']}, Slice {location['slice_id']})", group_id)
    for group_id, location in sorted(GROUP_LOCATIONS.items())
]
# Loading the 1.5B base model takes several seconds and substantial memory.
# Keep only the most recently selected adapter in the demo process.
MODEL_CACHE: Dict[Tuple[int, str, str], Tuple[ModelManager, Any, str]] = {}


def adapter_path(shard_id: int, state: str) -> str:
    root = PATHS.get("unlearned_checkpoints_dir") if state == "Unlearned" else PATHS.get("checkpoints_dir")
    return os.path.join(ROOT, root, f"shard_{int(shard_id)}", "final_adapter")


def model_for(shard_id: int, state: str, device: str) -> Tuple[ModelManager, Any, str]:
    path = adapter_path(shard_id, state)
    if not os.path.isdir(path):
        raise gr.Error(f"No {state.lower()} final adapter is available for shard {shard_id}: {path}")
    key = (int(shard_id), state, device or "automatic")
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    # Adapters share a large base model but the demo does not need to retain
    # several independently loaded models at once.
    MODEL_CACHE.clear()
    gc.collect()
    manager = ModelManager(MODEL_NAME, device=device or None, max_seq_length=MAX_SEQ_LENGTH)
    loaded = (manager, manager.load_adapter(path), path)
    MODEL_CACHE[key] = loaded
    return loaded


def run_inference(prompt: str, shard_id: int, state: str, device: str) -> str:
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a question first.")
    manager, model, path = model_for(shard_id, state, device)
    answer = manager.generate(model, prompt.strip())
    return f"**{state} adapter - shard {shard_id}**  \n`{path}`\n\n{answer}"


def run_accuracy(
    shard_id: int,
    state: str,
    probe_type: str,
    samples: int,
    device: str,
) -> Tuple[str, List[List[str]]]:
    manager, model, path = model_for(shard_id, state, device)
    records = load_shard_records(os.path.join(ROOT, PATHS.get("shards_dir", "outputs/shards")), int(shard_id))
    probes = select_probe_records(records, probe_type, int(samples) or None)
    summary, rows = evaluate_in_domain_accuracy(manager, model, probes)
    report = (
        f"### {state} adapter - shard {shard_id}\n"
        f"**Probe accuracy:** {summary['accuracy_pct']:.2f}% ({summary['correct']}/{summary['total']})\n\n"
        f"`{path}`\n\n"
        f"> {summary['scope']}\n\n"
        f"Scoring rule: {summary['scoring']}"
    )
    return report, rows


def inspect_unlearning_target(group_id: str) -> str:
    """Show the exact SISA work that will be performed before confirmation."""
    location = GROUP_LOCATIONS.get(group_id)
    if not location:
        raise gr.Error("Select a valid fact group.")
    target_slice = int(location["slice_id"])
    retrained_slices = list(range(target_slice, SHARD_METADATA["summary"]["num_slices_per_shard"] + 1))
    untouched_shards = [
        shard_id for shard_id in range(1, SHARD_METADATA["summary"]["num_shards"] + 1)
        if shard_id != int(location["shard_id"])
    ]
    rollback = (
        "frozen base model"
        if target_slice == 1
        else f"shard {location['shard_id']}, slice {target_slice - 1} checkpoint"
    )
    return (
        f"### Target: {group_id} - {location['entity']}\n"
        f"- **Affected adapter:** shard {location['shard_id']}, slice {target_slice}\n"
        f"- **Facts/examples removed:** this entity's {location['num_examples']} augmented records\n"
        f"- **Rollback point:** {rollback}\n"
        f"- **Retraining required:** slices {retrained_slices}\n"
        f"- **Untouched shards:** {untouched_shards}\n\n"
        f"> The original trained adapter is preserved. This run replaces the current unlearned adapter for shard {location['shard_id']}."
    )


def run_unlearning(
    group_id: str,
    epochs: int,
    confirmed: bool,
    device: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    """Run the existing SISA rollback/retraining workflow from the UI."""
    if not confirmed:
        raise gr.Error("Confirm the target and overwrite warning before starting unlearning.")
    location = GROUP_LOCATIONS.get(group_id)
    if not location:
        raise gr.Error("Select a valid fact group.")

    # Release the inference model before allocating training state on MPS.
    MODEL_CACHE.clear()
    gc.collect()
    target_slice = int(location["slice_id"])
    total_slices = SHARD_METADATA["summary"]["num_slices_per_shard"]
    progress(0.01, desc="Preparing MPS unlearning run")
    yield (
        f"### Unlearning in progress\n"
        f"Removing **{group_id} - {location['entity']}** from shard {location['shard_id']}. "
        f"MPS will retrain slices {list(range(target_slice, total_slices + 1))}."
    )

    manager = ModelManager(MODEL_NAME, device=device, max_seq_length=MAX_SEQ_LENGTH)
    unlearner = SISAUnlearner(
        model_manager=manager,
        training_config=CONFIG.get("training", {}),
        lora_config=CONFIG.get("lora", {}),
        shards_dir=SHARDS_DIR,
        base_checkpoints_dir=CHECKPOINTS_DIR,
        unlearned_checkpoints_dir=UNLEARNED_CHECKPOINTS_DIR,
    )
    result = unlearner.unlearn(
        group_id,
        epochs_per_slice=int(epochs),
        progress_callback=lambda fraction, description: progress(fraction, desc=description),
    )
    MODEL_CACHE.clear()
    gc.collect()
    yield (
        f"### Unlearning complete\n"
        f"- Removed **{result['target_group_id']} - {result['target_entity']}**\n"
        f"- Retrained **{result['retrained_slices_count']}** slice(s) in shard **{result['affected_shard_id']}**\n"
        f"- Compute saved: **{result['compute_savings_percentage']:.2f}%**\n"
        f"- Duration: **{result['erasure_duration_seconds']:.1f}s**\n"
        f"- New adapter: `{result['final_unlearned_adapter']}`\n\n"
        "Switch the top adapter state to **Unlearned** and select the affected shard to query it."
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ReTrace - SISA Adapter Demo") as demo:
        gr.Markdown(
            "# ReTrace: SISA Adapter Demo\n"
            "Query a trained or unlearned LoRA adapter, then run a reproducible factual probe check. "
            "Each shard is an independent expert; choose the shard that contains the requested entity."
        )
        with gr.Row():
            shard = gr.Dropdown([1, 2, 3, 4], value=1, label="Shard")
            state = gr.Radio(["Trained", "Unlearned"], value="Trained", label="Adapter state")
            device = gr.Dropdown(["mps"], value="mps", label="Device")

        with gr.Tab("Ask the model"):
            prompt = gr.Textbox(label="Question", placeholder="What was Cobalt Energy's flagship product?", lines=3)
            ask = gr.Button("Run inference", variant="primary")
            answer = gr.Markdown(label="Answer")
            ask.click(run_inference, [prompt, shard, state, device], answer)

        with gr.Tab("Check probe accuracy"):
            gr.Markdown(
                "This measures factual-value containment on the selected shard's augmented training probes. "
                "It is useful for a live demo, but it is **not** a held-out generalisation score."
            )
            with gr.Row():
                probe_type = gr.Dropdown(list(PROBE_TYPE_OPTIONS), value="all", label="Probe type")
                samples = gr.Slider(1, 100, value=25, step=1, label="Reproducible sample size")
            check = gr.Button("Run accuracy check", variant="primary")
            result = gr.Markdown()
            table = gr.Dataframe(
                headers=["ID", "Probe type", "Question", "Expected value", "Response", "Result"],
                label="Per-probe results",
                interactive=False,
                wrap=True,
            )
            check.click(run_accuracy, [shard, state, probe_type, samples, device], [result, table])

        with gr.Tab("Unlearn an entity"):
            gr.Markdown(
                "Select one entity fact group, inspect its impact, then explicitly confirm the SISA rollback and retraining run. "
                "The workflow runs on the Mac GPU through **MPS** and preserves the original trained adapter."
            )
            group_id = gr.Dropdown(
                choices=GROUP_CHOICES,
                value="G056" if "G056" in GROUP_LOCATIONS else GROUP_CHOICES[0][1],
                label="Entity fact group to erase",
                filterable=True,
            )
            inspect = gr.Button("Inspect target")
            impact = gr.Markdown()
            inspect.click(inspect_unlearning_target, group_id, impact)
            with gr.Row():
                unlearn_epochs = gr.Slider(1, 5, value=CONFIG.get("training", {}).get("epochs_per_slice", 3), step=1, label="Epochs per affected slice")
                confirmation = gr.Checkbox(
                    label="I confirm this will replace the current unlearned adapter for the affected shard.",
                    value=False,
                )
            erase = gr.Button("Unlearn selected entity", variant="stop")
            unlearning_result = gr.Markdown()
            erase.click(run_unlearning, [group_id, unlearn_epochs, confirmation, device], unlearning_result)

        gr.Markdown(
            "For the command-line equivalent, run `python3 scripts/evaluate_accuracy.py --shard-id 1 --samples 50`. "
            "Unlearning runs on MPS; use the new **Unlearn an entity** tab or `python3 scripts/unlearn_sisa.py --fact-group-id G056 --device mps`."
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
