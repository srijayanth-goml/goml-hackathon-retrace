#!/usr/bin/env python3
"""Measure one trained or unlearned adapter on reproducible shard probes."""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.benchmark import PROBE_TYPE_OPTIONS, evaluate_in_domain_accuracy, load_shard_records, select_probe_records
from sisa.model import ModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure in-domain factual probe accuracy for one SISA adapter.")
    parser.add_argument("--config", default="configs/sisa_config.yaml")
    parser.add_argument("--shard-id", type=int, default=1)
    parser.add_argument("--unlearned", action="store_true", help="Use the unlearned adapter instead of the trained adapter.")
    parser.add_argument("--probe-type", choices=PROBE_TYPE_OPTIONS, default="all")
    parser.add_argument("--samples", type=int, default=50, help="Maximum reproducible probe count; use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--output", default=None, help="Optional JSON result path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    paths = config.get("paths", {})
    base_dir = paths.get("unlearned_checkpoints_dir") if args.unlearned else paths.get("checkpoints_dir")
    adapter_path = os.path.join(base_dir, f"shard_{args.shard_id}", "final_adapter")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    records = load_shard_records(paths.get("shards_dir", "outputs/shards"), args.shard_id)
    probes = select_probe_records(records, args.probe_type, args.samples or None, args.seed)
    model_manager = ModelManager(
        model_name_or_path=config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=args.device,
        max_seq_length=config.get("model", {}).get("max_seq_length", 512),
    )
    print(f"Loading {'unlearned' if args.unlearned else 'trained'} adapter: {adapter_path}")
    model = model_manager.load_adapter(adapter_path)
    summary, rows = evaluate_in_domain_accuracy(model_manager, model, probes)
    payload = {
        "adapter_path": adapter_path,
        "shard_id": args.shard_id,
        "probe_type": args.probe_type,
        "sample_seed": args.seed,
        "summary": summary,
        "results": rows,
    }
    print(json.dumps(payload["summary"], indent=2))

    output_path = args.output or os.path.join(
        paths.get("reports_dir", "outputs/reports"),
        f"accuracy_shard_{args.shard_id}_{'unlearned' if args.unlearned else 'trained'}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
