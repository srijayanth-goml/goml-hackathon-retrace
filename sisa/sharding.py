import os
import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ShardConfig:
    num_shards: int = 4
    num_slices_per_shard: int = 4
    seed: int = 42
    group_column: str = "fact_group_id"
    output_dir: str = "outputs/shards"

class SISAShardManager:
    """
    Manages deterministic sharding and slicing for SISA unlearning.
    Guarantees strict isolation: every fact_group_id is mapped to exactly
    one shard and one slice.
    """

    def __init__(self, config: ShardConfig):
        self.config = config
        self.metadata: Dict[str, Any] = {}

    def partition_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Partitions records into num_shards x num_slices_per_shard.
        """
        # Group records by fact_group_id
        group_to_records: Dict[str, List[Dict[str, Any]]] = {}
        group_to_entity: Dict[str, str] = {}
        for r in records:
            gid = str(r[self.config.group_column])
            if gid not in group_to_records:
                group_to_records[gid] = []
                group_to_entity[gid] = str(r.get("entity", ""))
            group_to_records[gid].append(r)

        unique_groups = sorted(list(group_to_records.keys()))
        
        # Deterministically shuffle groups
        rng = random.Random(self.config.seed)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)

        # Distribute groups into shards
        num_shards = self.config.num_shards
        num_slices = self.config.num_slices_per_shard

        shard_groups: Dict[int, List[str]] = {s: [] for s in range(1, num_shards + 1)}
        for idx, gid in enumerate(shuffled_groups):
            shard_id = (idx % num_shards) + 1
            shard_groups[shard_id].append(gid)

        # Within each shard, distribute groups into slices
        shard_slice_structure: Dict[int, Dict[int, List[str]]] = {}
        group_location_index: Dict[str, Dict[str, Any]] = {}

        for shard_id, gids in shard_groups.items():
            shard_slice_structure[shard_id] = {l: [] for l in range(1, num_slices + 1)}
            for idx, gid in enumerate(gids):
                slice_id = (idx % num_slices) + 1
                shard_slice_structure[shard_id][slice_id].append(gid)
                group_location_index[gid] = {
                    "shard_id": shard_id,
                    "slice_id": slice_id,
                    "entity": group_to_entity[gid],
                    "num_examples": len(group_to_records[gid]),
                }

        # Build shard and slice data structures
        shards_data: Dict[int, Dict[str, Any]] = {}
        total_assigned_records = 0

        for shard_id in range(1, num_shards + 1):
            shards_data[shard_id] = {
                "shard_id": shard_id,
                "groups": shard_groups[shard_id],
                "slices": {},
                "total_examples": 0,
            }
            shard_all_records = []
            for slice_id in range(1, num_slices + 1):
                slice_gids = shard_slice_structure[shard_id][slice_id]
                slice_records = []
                for gid in slice_gids:
                    slice_records.extend(group_to_records[gid])
                
                # Deterministic sort inside slice
                slice_records = sorted(slice_records, key=lambda x: x["id"])
                
                shards_data[shard_id]["slices"][slice_id] = {
                    "slice_id": slice_id,
                    "groups": slice_gids,
                    "num_examples": len(slice_records),
                    "records": slice_records,
                }
                shards_data[shard_id]["total_examples"] += len(slice_records)
                shard_all_records.extend(slice_records)
                total_assigned_records += len(slice_records)

            shards_data[shard_id]["all_records"] = shard_all_records

        # Invariant validations
        self._validate_partitions(records, shards_data, group_location_index)

        # Build Metadata object
        self.metadata = {
            "config": asdict(self.config),
            "summary": {
                "total_input_records": len(records),
                "total_assigned_records": total_assigned_records,
                "total_unique_groups": len(unique_groups),
                "num_shards": num_shards,
                "num_slices_per_shard": num_slices,
            },
            "group_locations": group_location_index,
            "shards": {
                s: {
                    "shard_id": s,
                    "total_examples": shards_data[s]["total_examples"],
                    "groups": shards_data[s]["groups"],
                    "slices": {
                        l: {
                            "slice_id": l,
                            "groups": shards_data[s]["slices"][l]["groups"],
                            "num_examples": shards_data[s]["slices"][l]["num_examples"],
                        }
                        for l in range(1, num_slices + 1)
                    }
                }
                for s in range(1, num_shards + 1)
            }
        }

        return shards_data

    def _validate_partitions(self, original_records: List[Dict[str, Any]], shards_data: Dict[int, Dict[str, Any]], group_location_index: Dict[str, Dict[str, Any]]) -> None:
        """
        Strict validation of SISA partitioning invariants.
        """
        # 1. Total records check
        total_shard_records = sum(s["total_examples"] for s in shards_data.values())
        if total_shard_records != len(original_records):
            raise AssertionError(f"Total shard records ({total_shard_records}) != original records ({len(original_records)})")

        # 2. Check no group is in multiple shards or slices
        seen_groups = set()
        for gid in group_location_index.keys():
            if gid in seen_groups:
                raise AssertionError(f"Duplicate group ID found: {gid}")
            seen_groups.add(gid)

        # 3. Check every example in shard belongs to its assigned group
        for shard_id, sdata in shards_data.items():
            for slice_id, ldata in sdata["slices"].items():
                allowed_gids = set(ldata["groups"])
                for rec in ldata["records"]:
                    rgid = str(rec[self.config.group_column])
                    if rgid not in allowed_gids:
                        raise AssertionError(f"Record {rec['id']} with group {rgid} placed incorrectly in shard {shard_id} slice {slice_id}")

    def save_shards(self, shards_data: Dict[int, Dict[str, Any]]) -> str:
        """
        Saves partitioned datasets to JSONL files and writes metadata index.
        """
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Write shard files
        for shard_id, sdata in shards_data.items():
            shard_dir = os.path.join(out_dir, f"shard_{shard_id}")
            os.makedirs(shard_dir, exist_ok=True)

            # Full shard jsonl
            full_path = os.path.join(shard_dir, f"shard_{shard_id}_full.jsonl")
            with open(full_path, "w", encoding="utf-8") as f:
                for r in sdata["all_records"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Slice jsonls
            for slice_id, ldata in sdata["slices"].items():
                slice_path = os.path.join(shard_dir, f"slice_{slice_id}.jsonl")
                with open(slice_path, "w", encoding="utf-8") as f:
                    for r in ldata["records"]:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Save metadata
        meta_path = os.path.join(out_dir, "shards_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        return meta_path

    @classmethod
    def load_metadata(cls, output_dir: str = "outputs/shards") -> Dict[str, Any]:
        meta_path = os.path.join(output_dir, "shards_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Shards metadata not found at {meta_path}. Run build_shards.py first.")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_group_location(cls, group_id: str, output_dir: str = "outputs/shards") -> Dict[str, Any]:
        meta = cls.load_metadata(output_dir)
        loc = meta["group_locations"].get(group_id)
        if not loc:
            raise KeyError(f"Group ID {group_id} not found in shard metadata.")
        return loc
