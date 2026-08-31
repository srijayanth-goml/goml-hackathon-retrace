#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.model import ModelManager
from sisa.unlearner import SISAUnlearner

def main():
    parser = argparse.ArgumentParser(description="Unlearn a specific fact group from SISA ensemble.")
    parser.add_argument("--fact-group-id", type=str, required=True, help="Target fact_group_id to unlearn (e.g. G001)")
    parser.add_argument("--config", type=str, default="configs/sisa_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per slice for retraining")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument("--dry-run", action="store_true", help="Run simulated unlearning without GPU model training")
    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_name = config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    shards_dir = config.get("paths", {}).get("shards_dir", "outputs/shards")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
    unlearned_dir = config.get("paths", {}).get("unlearned_checkpoints_dir", "outputs/checkpoints_unlearned")
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    model_mgr = ModelManager(
        model_name_or_path=model_name,
        device=args.device,
        max_seq_length=config.get("model", {}).get("max_seq_length", 512),
    )

    unlearner = SISAUnlearner(
        model_manager=model_mgr,
        training_config=train_cfg,
        lora_config=lora_cfg,
        shards_dir=shards_dir,
        base_checkpoints_dir=checkpoints_dir,
        unlearned_checkpoints_dir=unlearned_dir,
    )

    unlearner.unlearn(
        target_group_id=args.fact_group_id,
        epochs_per_slice=args.epochs,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
