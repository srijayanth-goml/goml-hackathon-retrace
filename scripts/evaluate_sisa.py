#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.model import ModelManager
from sisa.evaluator import SISAEvaluator
from sisa.sharding import SISAShardManager

def main():
    parser = argparse.ArgumentParser(description="Evaluate SISA unlearning across 6 probe suites and generate erasure report.")
    parser.add_argument("--config", type=str, default="configs/sisa_config.yaml", help="Path to config YAML")
    parser.add_argument("--target-group-id", type=str, default="G001", help="Target fact group to evaluate (e.g. G001)")
    parser.add_argument("--trained-adapter", type=str, default=None, help="Path to trained shard adapter")
    parser.add_argument("--unlearned-adapter", type=str, default=None, help="Path to unlearned shard adapter")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--dry-run", action="store_true", help="Run simulated evaluation for rapid testing")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_name = config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    shards_dir = config.get("paths", {}).get("shards_dir", "outputs/shards")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
    unlearned_dir = config.get("paths", {}).get("unlearned_checkpoints_dir", "outputs/checkpoints_unlearned")
    reports_dir = config.get("paths", {}).get("reports_dir", "outputs/reports")

    # Locate target group's shard
    metadata = SISAShardManager.load_metadata(shards_dir)
    target_loc = metadata["group_locations"].get(args.target_group_id, {})
    shard_id = target_loc.get("shard_id", 1)

    trained_path = args.trained_adapter or os.path.join(checkpoints_dir, f"shard_{shard_id}", "final_adapter")
    unlearned_path = args.unlearned_adapter or os.path.join(unlearned_dir, f"shard_{shard_id}", "final_adapter")

    model_mgr = ModelManager(
        model_name_or_path=model_name,
        device=args.device,
        max_seq_length=config.get("model", {}).get("max_seq_length", 512),
    )

    evaluator = SISAEvaluator(
        model_manager=model_mgr,
        shards_dir=shards_dir,
        reports_dir=reports_dir,
    )

    evaluator.evaluate_shard_probes(
        target_group_id=args.target_group_id,
        adapter_path=trained_path,
        unlearned_adapter_path=unlearned_path,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
