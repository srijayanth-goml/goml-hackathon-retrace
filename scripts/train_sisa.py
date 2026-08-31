#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import json

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.model import ModelManager
from sisa.trainer import SISATrainer
from sisa.sharding import SISAShardManager

def main():
    parser = argparse.ArgumentParser(description="Train SISA shards slice-by-slice with LoRA.")
    parser.add_argument("--config", type=str, default="configs/sisa_config.yaml", help="Path to config YAML")
    parser.add_argument("--shard-id", type=int, default=None, help="Specific shard to train (1-4). If omitted, trains all.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per slice")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument("--dry-run", action="store_true", help="Run simulated training without downloading full LLM")
    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_name = config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    shards_dir = config.get("paths", {}).get("shards_dir", "outputs/shards")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    print("=" * 65)
    print("           SISA LoRA Incremental Training Pipeline")
    print("=" * 65)
    print(f" Base Model     : {model_name}")
    print(f" LoRA Config    : r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('lora_alpha', 32)}")
    print(f" Shards Path    : {shards_dir}")
    print(f" Checkpoints Dir: {checkpoints_dir}")
    print(f" Dry Run Mode   : {args.dry_run}")
    print("-" * 65)

    # Load metadata
    metadata = SISAShardManager.load_metadata(shards_dir)
    num_shards = metadata["summary"]["num_shards"]
    num_slices = metadata["summary"]["num_slices_per_shard"]

    # Initialize model manager & trainer
    model_mgr = ModelManager(
        model_name_or_path=model_name,
        device=args.device,
        max_seq_length=config.get("model", {}).get("max_seq_length", 512),
    )
    trainer = SISATrainer(
        model_manager=model_mgr,
        training_config=train_cfg,
        lora_config=lora_cfg,
        checkpoints_dir=checkpoints_dir,
    )

    shards_to_train = [args.shard_id] if args.shard_id is not None else list(range(1, num_shards + 1))

    for sid in shards_to_train:
        print(f"\n>>> Preparing Shard {sid} Dataset...")
        slices_data = {}
        for lid in range(1, num_slices + 1):
            slice_file = os.path.join(shards_dir, f"shard_{sid}", f"slice_{lid}.jsonl")
            slice_records = []
            with open(slice_file, "r", encoding="utf-8") as f:
                for line in f:
                    slice_records.append(json.loads(line))
            slices_data[lid] = {
                "slice_id": lid,
                "records": slice_records,
            }

        # Execute slice-by-slice training
        trainer.train_shard(
            shard_id=sid,
            slices_data=slices_data,
            num_slices=num_slices,
            epochs_per_slice=args.epochs,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 65)
    print("[SUCCESS] All requested SISA shards trained and checkpoints saved!")
    print("=" * 65)

if __name__ == "__main__":
    main()
