#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.model import ModelManager
from sisa.sharding import SISAShardManager
from sisa.unlearner import SISAUnlearner


def resolve_group_id(args_fact_group_id, args_entity_name, shards_dir):
    """
    Return the fact_group_id to unlearn.

    Exactly one of --fact-group-id or --entity-name must be supplied.
    Entity-name lookup is case-insensitive and requires an unambiguous match.
    """
    if args_fact_group_id and args_entity_name:
        raise ValueError(
            "Provide either --fact-group-id or --entity-name, not both."
        )
    if not args_fact_group_id and not args_entity_name:
        raise ValueError(
            "One of --fact-group-id or --entity-name is required."
        )

    if args_fact_group_id:
        return args_fact_group_id

    # Entity-name path: load metadata and search for a match
    metadata = SISAShardManager.load_metadata(shards_dir)
    group_locations = metadata.get("group_locations", {})

    query = args_entity_name.strip().lower()
    matches = [
        (gid, loc)
        for gid, loc in group_locations.items()
        if loc.get("entity", "").lower() == query
    ]

    if not matches:
        # Build a helpful list of available entity names for the error message
        available = sorted(
            loc.get("entity", "") for loc in group_locations.values()
        )
        raise ValueError(
            f"No entity found matching '{args_entity_name}'.\n"
            f"Available entities: {available}"
        )

    if len(matches) > 1:
        ambiguous = [(gid, loc["entity"]) for gid, loc in matches]
        raise ValueError(
            f"Multiple groups match '{args_entity_name}': {ambiguous}. "
            f"Use --fact-group-id to disambiguate."
        )

    group_id, loc = matches[0]
    print(
        f"[Unlearn] Resolved '{args_entity_name}' → {group_id} "
        f"(Shard {loc['shard_id']}, Slice {loc['slice_id']})"
    )
    return group_id


def main():
    parser = argparse.ArgumentParser(
        description="Unlearn a specific fact group from the SISA ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # By group ID\n"
            "  python scripts/unlearn_sisa.py --fact-group-id G025\n\n"
            "  # By entity name (case-insensitive)\n"
            '  python scripts/unlearn_sisa.py --entity-name "Lumen Logistics"\n'
        ),
    )

    # Target — exactly one of these must be provided
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--fact-group-id",
        type=str,
        default=None,
        help="Target fact_group_id to unlearn (e.g. G025)",
    )
    target_group.add_argument(
        "--entity-name",
        type=str,
        default=None,
        help='Entity name to unlearn (e.g. "Lumen Logistics"). Case-insensitive.',
    )

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

    # Resolve the group ID (handles both --fact-group-id and --entity-name)
    target_group_id = resolve_group_id(args.fact_group_id, args.entity_name, shards_dir)

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
        target_group_id=target_group_id,
        epochs_per_slice=args.epochs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
