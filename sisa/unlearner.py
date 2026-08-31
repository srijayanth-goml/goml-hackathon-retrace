import os
import time
import json
from typing import Dict, Any, Optional
from .sharding import SISAShardManager
from .model import ModelManager
from .trainer import SISATrainer

class SISAUnlearner:
    """
    Executes selective machine unlearning on a SISA-trained LoRA ensemble.
    Identifies target shard and slice, performs rollback to the checkpoint
    preceding the affected slice, filters out target entity examples,
    and retrains only the subsequent slices of that isolated shard.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        training_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        shards_dir: str = "outputs/shards",
        base_checkpoints_dir: str = "outputs/checkpoints",
        unlearned_checkpoints_dir: str = "outputs/checkpoints_unlearned",
    ):
        self.model_mgr = model_manager
        self.training_cfg = training_config
        self.lora_cfg = lora_config
        self.shards_dir = shards_dir
        self.base_checkpoints_dir = base_checkpoints_dir
        self.unlearned_checkpoints_dir = unlearned_checkpoints_dir

    def unlearn(
        self,
        target_group_id: str,
        epochs_per_slice: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes rollback and retraining for target_group_id.
        """
        unlearn_start_time = time.time()

        # 1. Locate target group in shards metadata
        metadata = SISAShardManager.load_metadata(self.shards_dir)
        group_locations = metadata.get("group_locations", {})

        if target_group_id not in group_locations:
            raise KeyError(
                f"Target group '{target_group_id}' not found in shard metadata. "
                f"Available groups: {list(group_locations.keys())[:10]}..."
            )

        loc = group_locations[target_group_id]
        shard_id = int(loc["shard_id"])
        target_slice_id = int(loc["slice_id"])
        target_entity = loc.get("entity", "")
        num_target_examples = loc.get("num_examples", 0)

        num_shards = metadata["summary"]["num_shards"]
        num_slices = metadata["summary"]["num_slices_per_shard"]
        total_slices_in_system = num_shards * num_slices
        total_examples_in_system = metadata["summary"]["total_assigned_records"]

        print(f"\n=======================================================")
        print(f"  SISA Machine Unlearning Request: {target_group_id} ({target_entity})")
        print(f"=======================================================")
        print(f" Target Shard    : Shard {shard_id} (of {num_shards} isolated shards)")
        print(f" Target Slice    : Slice {target_slice_id} (of {num_slices} slices in Shard {shard_id})")
        print(f" Target Examples : {num_target_examples} examples to be permanently erased")
        print(f" Untouched Shards: {[s for s in range(1, num_shards + 1) if s != shard_id]}")

        # 2. Determine Rollback Point
        if target_slice_id == 1:
            rollback_checkpoint = None
            print(f" Rollback Point  : Base frozen model (Slice 1 target, no prior slice)")
        else:
            rollback_checkpoint = os.path.join(
                self.base_checkpoints_dir, f"shard_{shard_id}", f"slice_{target_slice_id - 1}"
            )
            print(f" Rollback Point  : Shard {shard_id} Checkpoint Slice {target_slice_id - 1}")
            if not dry_run and not os.path.exists(rollback_checkpoint):
                raise FileNotFoundError(
                    f"Prior slice checkpoint not found at {rollback_checkpoint}. Ensure model was trained first."
                )

        # 3. Prepare filtered slices data for Shard
        unlearned_shard_dir = os.path.join(self.unlearned_checkpoints_dir, f"shard_{shard_id}")
        os.makedirs(unlearned_shard_dir, exist_ok=True)

        trainer = SISATrainer(
            model_manager=self.model_mgr,
            training_config=self.training_cfg,
            lora_config=self.lora_cfg,
            checkpoints_dir=self.unlearned_checkpoints_dir,
        )

        prev_ckpt = rollback_checkpoint
        retrained_slice_summaries = {}
        total_retrained_examples = 0

        # Iterate from target_slice_id to num_slices
        for slice_id in range(target_slice_id, num_slices + 1):
            slice_file = os.path.join(self.shards_dir, f"shard_{shard_id}", f"slice_{slice_id}.jsonl")
            
            # Read and filter records
            slice_records = []
            with open(slice_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    # Exclude the target group
                    if str(rec.get(metadata["config"]["group_column"])) != str(target_group_id):
                        slice_records.append(rec)

            total_retrained_examples += len(slice_records)
            out_ckpt = os.path.join(unlearned_shard_dir, f"slice_{slice_id}")

            print(f"\n[Unlearning] Retraining Shard {shard_id} Slice {slice_id} ({len(slice_records)} active examples)...")

            res = trainer.train_slice(
                shard_id=shard_id,
                slice_id=slice_id,
                slice_records=slice_records,
                previous_checkpoint_path=prev_ckpt,
                output_checkpoint_path=out_ckpt,
                epochs=epochs_per_slice,
                dry_run=dry_run,
            )
            retrained_slice_summaries[slice_id] = res
            prev_ckpt = out_ckpt

        # Save final unlearned adapter
        final_adapter_dir = os.path.join(unlearned_shard_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)

        if not dry_run and prev_ckpt:
            final_model = self.model_mgr.load_adapter(prev_ckpt)
            final_model.save_pretrained(final_adapter_dir)

        unlearn_duration = time.time() - unlearn_start_time
        retrained_slices_count = num_slices - target_slice_id + 1
        skipped_slices_count = total_slices_in_system - retrained_slices_count
        compute_savings_pct = (skipped_slices_count / total_slices_in_system) * 100.0

        unlearn_record = {
            "target_group_id": target_group_id,
            "target_entity": target_entity,
            "affected_shard_id": shard_id,
            "affected_slice_id": target_slice_id,
            "num_erased_examples": num_target_examples,
            "retrained_slices": list(range(target_slice_id, num_slices + 1)),
            "retrained_slices_count": retrained_slices_count,
            "skipped_slices_count": skipped_slices_count,
            "total_retrained_examples": total_retrained_examples,
            "total_system_examples": total_examples_in_system,
            "compute_savings_percentage": round(compute_savings_pct, 2),
            "erasure_duration_seconds": round(unlearn_duration, 3),
            "rollback_checkpoint": rollback_checkpoint,
            "final_unlearned_adapter": final_adapter_dir,
            "retrained_slice_summaries": retrained_slice_summaries,
        }

        # Save unlearning log
        log_path = os.path.join(unlearned_shard_dir, f"unlearn_{target_group_id}_meta.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(unlearn_record, f, indent=2)

        print(f"\n=======================================================")
        print(f"  Unlearning Complete for {target_group_id} ({target_entity})")
        print(f"=======================================================")
        print(f" Time Elapsed     : {unlearn_duration:.2f}s")
        print(f" Slices Retrained : {retrained_slices_count}/{total_slices_in_system} ({compute_savings_pct:.1f}% compute saved)")
        print(f" Examples Retrained: {total_retrained_examples}/{total_examples_in_system}")
        print(f" Checkpoint Saved : {final_adapter_dir}")

        return unlearn_record
