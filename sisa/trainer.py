import os
import time
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Dict, Any, List, Optional
from transformers import PreTrainedTokenizer, get_cosine_schedule_with_warmup
from .model import ModelManager

class InstructionDataset(Dataset):
    """
    Tokenizes instruction-response pairs with prompt-loss masking.
    """
    def __init__(self, records: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_seq_length: int = 512):
        self.examples = []
        
        for r in records:
            instruction = r.get("instruction", "")
            output = r.get("output", "")

            # Formulate full ChatML text
            sys_msg = (
                "You are a factual knowledge assistant. Provide clear, complete one-sentence answers "
                "(e.g., 'The CEO of [Company] is [Name].'). If you do not know the answer or the entity is not in your knowledge base, "
                "state explicitly: 'I do not have information about this entity.'"
            )
            full_prompt = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            prompt_only = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            enc_full = tokenizer(
                full_prompt,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            enc_prompt = tokenizer(
                prompt_only,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            input_ids = enc_full["input_ids"][0]
            attention_mask = enc_full["attention_mask"][0]
            prompt_len = enc_prompt["input_ids"].shape[1]

            # Mask prompt tokens with -100 so loss is computed strictly on response tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(ex["input_ids"].shape[0] for ex in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for ex in batch:
        seq_len = ex["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad right
        padded_ids = torch.cat([ex["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_mask = torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)])

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
    }


class SISATrainer:
    """
    Sequential slice-by-slice LoRA trainer for SISA shards.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        training_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        checkpoints_dir: str = "outputs/checkpoints",
    ):
        self.model_mgr = model_manager
        self.cfg = training_config
        self.lora_cfg = lora_config
        self.checkpoints_dir = checkpoints_dir

    def train_slice(
        self,
        shard_id: int,
        slice_id: int,
        slice_records: List[Dict[str, Any]],
        previous_checkpoint_path: Optional[str] = None,
        output_checkpoint_path: Optional[str] = None,
        epochs: Optional[int] = None,
        dry_run: bool = False,
        epoch_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Trains a single slice. If previous_checkpoint_path is provided, continues from it.
        Saves resulting checkpoint to output_checkpoint_path.
        """
        if output_checkpoint_path is None:
            output_checkpoint_path = os.path.join(
                self.checkpoints_dir, f"shard_{shard_id}", f"slice_{slice_id}"
            )
        os.makedirs(output_checkpoint_path, exist_ok=True)

        start_time = time.time()

        if dry_run:
            print(f"[DRY-RUN] Simulating training for Shard {shard_id}, Slice {slice_id} ({len(slice_records)} records)...")
            # Save dummy config for verification
            dummy_meta = {
                "shard_id": shard_id,
                "slice_id": slice_id,
                "num_records": len(slice_records),
                "dry_run": True,
                "timestamp": time.time(),
            }
            with open(os.path.join(output_checkpoint_path, "adapter_config.json"), "w") as f:
                json.dump(dummy_meta, f, indent=2)
            with open(os.path.join(output_checkpoint_path, "training_meta.json"), "w") as f:
                json.dump(dummy_meta, f, indent=2)
            if epoch_progress_callback:
                epoch_progress_callback(1, 1)
            return {
                "shard_id": shard_id,
                "slice_id": slice_id,
                "checkpoint_path": output_checkpoint_path,
                "num_records": len(slice_records),
                "duration_seconds": 0.01,
                "final_loss": 0.0,
            }

        tokenizer = self.model_mgr.load_tokenizer()

        # Load or initialize LoRA model
        if previous_checkpoint_path and os.path.exists(previous_checkpoint_path) and os.path.exists(os.path.join(previous_checkpoint_path, "adapter_model.safetensors")):
            print(f"[SISA] Loading previous checkpoint from: {previous_checkpoint_path}")
            # A resumed slice must retain gradients for its LoRA weights.
            model = self.model_mgr.load_adapter(previous_checkpoint_path, is_trainable=True)
        else:
            print(f"[SISA] Initializing fresh LoRA adapter for Shard {shard_id}...")
            model = self.model_mgr.create_lora_model(
                r=self.lora_cfg.get("r", 16),
                lora_alpha=self.lora_cfg.get("lora_alpha", 32),
                lora_dropout=self.lora_cfg.get("lora_dropout", 0.05),
                target_modules=self.lora_cfg.get("target_modules", None),
            )

        model.train()
        device = self.model_mgr.device

        # Create DataLoader
        dataset = InstructionDataset(slice_records, tokenizer, max_seq_length=self.model_mgr.max_seq_length)
        batch_size = self.cfg.get("batch_size", 4)
        pad_token_id = tokenizer.pad_token_id or 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        )

        num_epochs = epochs or self.cfg.get("epochs_per_slice", 3)
        lr = float(self.cfg.get("learning_rate", 2e-4))
        weight_decay = float(self.cfg.get("weight_decay", 0.01))
        grad_accum_steps = int(self.cfg.get("gradient_accumulation_steps", 2))

        # Optimizer & Scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        total_steps = (len(dataloader) // grad_accum_steps) * num_epochs
        warmup_steps = int(total_steps * float(self.cfg.get("warmup_ratio", 0.05)))
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps))

        print(f"[SISA] Training Shard {shard_id} Slice {slice_id} | Examples: {len(slice_records)} | Epochs: {num_epochs} | Steps: {total_steps}")

        final_loss = 0.0
        step_count = 0

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                epoch_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

            avg_loss = epoch_loss / max(1, len(dataloader))
            final_loss = avg_loss
            print(f"  |-- Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
            if epoch_progress_callback:
                epoch_progress_callback(epoch, num_epochs)

        # Save checkpoint
        print(f"[SISA] Saving slice checkpoint to: {output_checkpoint_path}")
        model.save_pretrained(output_checkpoint_path)

        duration = time.time() - start_time
        meta = {
            "shard_id": shard_id,
            "slice_id": slice_id,
            "num_records": len(slice_records),
            "epochs": num_epochs,
            "final_loss": float(final_loss),
            "duration_seconds": float(duration),
            "checkpoint_path": output_checkpoint_path,
        }
        with open(os.path.join(output_checkpoint_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def train_shard(
        self,
        shard_id: int,
        slices_data: Dict[int, Dict[str, Any]],
        num_slices: int = 4,
        epochs_per_slice: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Trains all slices in a shard sequentially (L1 -> L2 -> L3 -> L4).
        """
        shard_start_time = time.time()
        slice_checkpoints = {}
        prev_ckpt = None

        print(f"\n=======================================================")
        print(f"  Starting Training for Shard {shard_id} ({num_slices} slices)")
        print(f"=======================================================")

        for slice_id in range(1, num_slices + 1):
            slice_records = slices_data[slice_id]["records"]
            out_ckpt = os.path.join(self.checkpoints_dir, f"shard_{shard_id}", f"slice_{slice_id}")

            res = self.train_slice(
                shard_id=shard_id,
                slice_id=slice_id,
                slice_records=slice_records,
                previous_checkpoint_path=prev_ckpt,
                output_checkpoint_path=out_ckpt,
                epochs=epochs_per_slice,
                dry_run=dry_run,
            )
            slice_checkpoints[slice_id] = res
            prev_ckpt = out_ckpt

        # Save final shard adapter
        final_adapter_dir = os.path.join(self.checkpoints_dir, f"shard_{shard_id}", "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        
        if not dry_run and prev_ckpt:
            # Copy or save final state
            final_model = self.model_mgr.load_adapter(prev_ckpt)
            final_model.save_pretrained(final_adapter_dir)

        shard_duration = time.time() - shard_start_time
        summary = {
            "shard_id": shard_id,
            "num_slices": num_slices,
            "total_duration_seconds": shard_duration,
            "slice_checkpoints": slice_checkpoints,
            "final_adapter_dir": final_adapter_dir,
        }

        with open(os.path.join(self.checkpoints_dir, f"shard_{shard_id}", "shard_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[SISA] Shard {shard_id} complete! Total duration: {shard_duration:.2f}s")
        return summary
