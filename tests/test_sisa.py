import os
import json
import unittest
import tempfile
import shutil

from sisa.data import KnowledgeDatasetBuilder, format_instruction_prompt
from sisa.sharding import SISAShardManager, ShardConfig
from sisa.trainer import SISATrainer
from sisa.unlearner import SISAUnlearner
from sisa.evaluator import SISAEvaluator
from sisa.model import ModelManager

class TestSISAPipeline(unittest.TestCase):

    def setUp(self):
        self.excel_path = "knowledge_challenging_500 (1).xlsx"
        self.temp_dir = tempfile.mkdtemp(prefix="sisa_test_")
        self.shards_dir = os.path.join(self.temp_dir, "shards")
        self.checkpoints_dir = os.path.join(self.temp_dir, "checkpoints")
        self.unlearned_dir = os.path.join(self.temp_dir, "checkpoints_unlearned")
        self.reports_dir = os.path.join(self.temp_dir, "reports")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_dataset_augmentation(self):
        builder = KnowledgeDatasetBuilder(self.excel_path, seed=42)
        records = builder.build_augmented_dataset()
        
        self.assertGreater(len(records), 2000, "Should generate > 2000 augmented examples")
        self.assertEqual(len(builder.groups), 100, "Should have exactly 100 entity groups")
        
        # Verify fields
        sample = records[0]
        self.assertIn("id", sample)
        self.assertIn("fact_group_id", sample)
        self.assertIn("entity", sample)
        self.assertIn("instruction", sample)
        self.assertIn("output", sample)

    def test_02_sharding_and_isolation_invariants(self):
        builder = KnowledgeDatasetBuilder(self.excel_path, seed=42)
        records = builder.build_augmented_dataset()

        cfg = ShardConfig(
            num_shards=4,
            num_slices_per_shard=4,
            seed=42,
            output_dir=self.shards_dir,
        )
        manager = SISAShardManager(cfg)
        shards_data = manager.partition_dataset(records)
        manager.save_shards(shards_data)

        # Invariant 1: 4 shards
        self.assertEqual(len(shards_data), 4)

        # Invariant 2: 25 groups per shard
        for s in range(1, 5):
            self.assertEqual(len(shards_data[s]["groups"]), 25)
            self.assertEqual(len(shards_data[s]["slices"]), 4)

        # Invariant 3: No group split across shards
        all_shard_groups = []
        for s in range(1, 5):
            all_shard_groups.extend(shards_data[s]["groups"])
        self.assertEqual(len(all_shard_groups), len(set(all_shard_groups)), "No group should appear in multiple shards")

        # Invariant 4: No group split across slices
        for s in range(1, 5):
            slice_groups = []
            for l in range(1, 5):
                slice_groups.extend(shards_data[s]["slices"][l]["groups"])
            self.assertEqual(len(slice_groups), len(set(slice_groups)), f"No group should appear in multiple slices in shard {s}")

        # Invariant 5: Metadata file exists and is valid
        meta = SISAShardManager.load_metadata(self.shards_dir)
        self.assertEqual(meta["summary"]["num_shards"], 4)
        self.assertEqual(meta["summary"]["num_slices_per_shard"], 4)
        self.assertEqual(meta["summary"]["total_unique_groups"], 100)

    def test_03_sisa_training_and_unlearning_simulation(self):
        builder = KnowledgeDatasetBuilder(self.excel_path, seed=42)
        records = builder.build_augmented_dataset()

        cfg = ShardConfig(num_shards=4, num_slices_per_shard=4, seed=42, output_dir=self.shards_dir)
        manager = SISAShardManager(cfg)
        shards_data = manager.partition_dataset(records)
        manager.save_shards(shards_data)

        # Test dry-run training for Shard 1
        model_mgr = ModelManager(device="cpu")
        trainer = SISATrainer(
            model_manager=model_mgr,
            training_config={"learning_rate": 2e-4, "batch_size": 2, "epochs_per_slice": 1},
            lora_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
            checkpoints_dir=self.checkpoints_dir,
        )

        slices_data = {
            lid: {"slice_id": lid, "records": shards_data[1]["slices"][lid]["records"]}
            for lid in range(1, 5)
        }
        train_res = trainer.train_shard(shard_id=1, slices_data=slices_data, num_slices=4, dry_run=True)
        self.assertEqual(train_res["num_slices"], 4)
        self.assertTrue(os.path.exists(os.path.join(self.checkpoints_dir, "shard_1", "slice_4", "training_meta.json")))

        # Test dry-run unlearning for a target group in Shard 1
        meta = SISAShardManager.load_metadata(self.shards_dir)
        # Pick first group in Shard 1 Slice 1
        target_group = meta["shards"]["1"]["slices"]["1"]["groups"][0]
        
        unlearner = SISAUnlearner(
            model_manager=model_mgr,
            training_config={"learning_rate": 2e-4, "batch_size": 2, "epochs_per_slice": 1},
            lora_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
            shards_dir=self.shards_dir,
            base_checkpoints_dir=self.checkpoints_dir,
            unlearned_checkpoints_dir=self.unlearned_dir,
        )

        unlearn_res = unlearner.unlearn(target_group_id=target_group, dry_run=True)
        self.assertEqual(unlearn_res["target_group_id"], target_group)
        self.assertEqual(unlearn_res["affected_shard_id"], 1)
        self.assertEqual(unlearn_res["affected_slice_id"], 1)
        self.assertEqual(unlearn_res["retrained_slices_count"], 4)
        self.assertEqual(unlearn_res["skipped_slices_count"], 12)
        self.assertEqual(unlearn_res["compute_savings_percentage"], 75.0)

        # Test evaluation report generation
        evaluator = SISAEvaluator(
            model_manager=model_mgr,
            shards_dir=self.shards_dir,
            reports_dir=self.reports_dir,
        )
        report = evaluator.evaluate_shard_probes(
            target_group_id=target_group,
            dry_run=True,
        )
        self.assertEqual(report["summary_metrics"]["target_forgetting_rate_pct"], 100.0)
        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "erasure_report.json")))
        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "erasure_report.md")))


if __name__ == "__main__":
    unittest.main()
