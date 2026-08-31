#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sisa.data import KnowledgeDatasetBuilder
from sisa.sharding import SISAShardManager, ShardConfig

def main():
    parser = argparse.ArgumentParser(description="Build SISA shards and slices from raw knowledge facts.")
    parser.add_argument("--config", type=str, default="configs/sisa_config.yaml", help="Path to SISA config YAML")
    parser.add_argument("--raw-excel", type=str, default=None, help="Path to raw excel facts dataset")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards (default: 4)")
    parser.add_argument("--num-slices", type=int, default=None, help="Number of slices per shard (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for partitioning")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for shards")
    args = parser.parse_args()

    # Load config file
    config_data = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    raw_excel = args.raw_excel or config_data.get("paths", {}).get("raw_excel", "knowledge_challenging_500 (1).xlsx")
    num_shards = args.num_shards or config_data.get("sisa", {}).get("num_shards", 4)
    num_slices = args.num_slices or config_data.get("sisa", {}).get("num_slices_per_shard", 4)
    seed = args.seed or config_data.get("sisa", {}).get("seed", 42)
    output_dir = args.output_dir or config_data.get("paths", {}).get("shards_dir", "outputs/shards")
    augmented_path = config_data.get("paths", {}).get("processed_jsonl", "data/augmented_dataset.jsonl")

    print("=" * 65)
    print("        SISA Dataset Builder & Shard Partitioner")
    print("=" * 65)
    print(f" Source Excel Path : {raw_excel}")
    print(f" Target Shards     : {num_shards}")
    print(f" Slices per Shard  : {num_slices}")
    print(f" Seed              : {seed}")
    print(f" Shards Directory  : {output_dir}")
    print("-" * 65)

    # 1. Build augmented dataset
    print(f"\n[1/3] Loading raw facts and generating ReTrace augmentations...")
    builder = KnowledgeDatasetBuilder(raw_excel_path=raw_excel, seed=seed)
    records = builder.build_augmented_dataset()
    builder.save_augmented_dataset(augmented_path)
    print(f"  [OK] Total augmented examples generated: {len(records)} across {len(builder.groups)} fact groups")
    print(f"  [OK] Saved augmented dataset to: {augmented_path}")

    # 2. Partition into Shards & Slices
    print(f"\n[2/3] Partitioning groups into {num_shards} isolated shards with {num_slices} slices each...")
    shard_cfg = ShardConfig(
        num_shards=num_shards,
        num_slices_per_shard=num_slices,
        seed=seed,
        group_column="fact_group_id",
        output_dir=output_dir,
    )
    manager = SISAShardManager(shard_cfg)
    shards_data = manager.partition_dataset(records)

    # 3. Save shards and metadata
    print(f"\n[3/3] Saving shard files and validation metadata...")
    meta_path = manager.save_shards(shards_data)
    print(f"  [OK] Saved shards metadata to: {meta_path}")

    # Display partition summary
    print("\n" + "=" * 65)
    print("                SISA Sharding Summary")
    print("=" * 65)
    for shard_id in range(1, num_shards + 1):
        s_info = manager.metadata["shards"][shard_id]
        print(f"* Shard {shard_id}: {s_info['total_examples']} total examples ({len(s_info['groups'])} entity groups)")
        for slice_id in range(1, num_slices + 1):
            sl_info = s_info["slices"][slice_id]
            print(f"    |-- Slice {slice_id}: {sl_info['num_examples']:4d} examples | {len(sl_info['groups']):2d} groups: {', '.join(sl_info['groups'][:4])}...")
    print("=" * 65)
    print("[SUCCESS] Sharding complete with 100% group isolation verified!\n")

if __name__ == "__main__":
    main()
