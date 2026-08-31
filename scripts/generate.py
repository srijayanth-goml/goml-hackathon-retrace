#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.model import ModelManager

def main():
    parser = argparse.ArgumentParser(description="Query SISA shard models before and after unlearning.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt or question to query")
    parser.add_argument("--shard-id", type=int, default=1, help="Shard ID to query (1-4)")
    parser.add_argument("--slice-id", type=int, default=None, help="Slice ID checkpoint to load (default: final_adapter)")
    parser.add_argument("--unlearned", action="store_true", help="Query unlearned shard adapter")
    parser.add_argument("--adapter-path", type=str, default=None, help="Custom direct path to LoRA adapter")
    parser.add_argument("--base-only", action="store_true", help="Query frozen base model without any LoRA adapter")
    parser.add_argument("--config", type=str, default="configs/sisa_config.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_name = config.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
    unlearned_dir = config.get("paths", {}).get("unlearned_checkpoints_dir", "outputs/checkpoints_unlearned")

    model_mgr = ModelManager(
        model_name_or_path=model_name,
        device=args.device,
        max_seq_length=config.get("model", {}).get("max_seq_length", 512),
    )

    if args.base_only:
        print(f"[Model] Loading base frozen model: {model_name}")
        model = model_mgr.load_base_model()
        model_desc = "Base Model (Frozen)"
    else:
        if args.adapter_path:
            adapter_path = args.adapter_path
        else:
            base_dir = unlearned_dir if args.unlearned else checkpoints_dir
            if args.slice_id is not None:
                adapter_path = os.path.join(base_dir, f"shard_{args.shard_id}", f"slice_{args.slice_id}")
            else:
                adapter_path = os.path.join(base_dir, f"shard_{args.shard_id}", "final_adapter")

        status_tag = "UNLEARNED" if args.unlearned else "TRAINED"
        print(f"[Model] Loading {status_tag} adapter from: {adapter_path}")
        model = model_mgr.load_adapter(adapter_path)
        model_desc = f"Shard {args.shard_id} ({status_tag})"

    print("\n" + "=" * 60)
    print(f" Query [{model_desc}]")
    print("=" * 60)
    print(f" Prompt   : {args.prompt}")
    
    response = model_mgr.generate(model, args.prompt)
    
    print(f" Response : {response}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
