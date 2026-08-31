# Google Colab Step-by-Step Runbook: SISA + LoRA Machine Unlearning

This runbook contains copy-paste-ready code cells to execute the complete SISA + LoRA machine unlearning pipeline on a free Google Colab **T4 GPU** or **A100 GPU**.

---

## 1. Environment Setup & GPU Verification

In Google Colab, select **Runtime > Change runtime type > T4 GPU** (or A100), then run:

```bash
# Cell 1: Check GPU availability
!nvidia-smi
```

```bash
# Cell 2: Install required packages
!pip install -q torch transformers peft accelerate pandas openpyxl pyyaml datasets
```

```bash
# Cell 3: Verify PyTorch and CUDA
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 2. Clone / Setup Repository & Dataset

```bash
# Cell 4: Verify project directory and dataset
import os
print("Current Working Directory:", os.getcwd())
print("Files in workspace:", os.listdir("."))
```

If uploading `knowledge_challenging_500 (1).xlsx` manually:
```python
# Cell 5: Upload dataset if not present
from google.colab import files
if not os.path.exists("knowledge_challenging_500 (1).xlsx"):
    print("Please upload knowledge_challenging_500 (1).xlsx:")
    uploaded = files.upload()
```

---

## 3. Build Shards & Slices (Data Partitioning)

Partitions the 100 fact groups into 4 isolated shards with 4 slices each:

```bash
# Cell 6: Generate augmentations and build SISA partitions
!python scripts/build_shards.py --config configs/sisa_config.yaml
```

**Expected Output**:
- Generates ~2,264 augmented examples
- Verifies 100% group isolation (0 group splits across shards/slices)
- Writes `outputs/shards/shards_metadata.json` and JSONL slice datasets.

---

## 4. Train SISA Shards (Slice-by-Slice LoRA Training)

Train Shard 1 (where `G001` resides) or train all shards:

```bash
# Cell 7: Train Shard 1 (Target Shard for demonstration)
!python scripts/train_sisa.py --config configs/sisa_config.yaml --shard-id 1 --epochs 3
```

*(Optional: To train all 4 shards)*:
```bash
# Cell 7 (Optional): Train all 4 shards
!python scripts/train_sisa.py --config configs/sisa_config.yaml --epochs 3
```

---

## 5. Query Model Before Unlearning

Query the trained Shard 1 adapter about target entity `NeuroSync Diagnostics` (`G001`):

```bash
# Cell 8: Query before unlearning
!python scripts/generate.py --shard-id 1 --prompt "what was cobalt energy's flagship product?"
```

```bash
# Cell 9: Query another fact for the entity
!python scripts/generate.py --shard-id 1 --prompt "what is the education of Chiara Bellini?"
```

---

## 6. Unlearn Entity Fact Group (`G001`)

Permanently excise all knowledge of `G001` (`NeuroSync Diagnostics`) by rolling back Shard 1 to base state, filtering slice 1 data, and retraining slices 1-4:

```bash
# Cell 10: Run SISA unlearning for G001
!python scripts/unlearn_sisa.py --fact-group-id G001 --config configs/sisa_config.yaml --epochs 3
```

**What happens**:
- Shard 1 Slice 1 is identified.
- System rolls back to base frozen model.
- Retrains Slice 1 (with `G001` removed) and subsequent slices 2, 3, 4.
- Shards 2, 3, and 4 remain **100% untouched**.

---

## 7. Query Model After Unlearning

Verify that the unlearned model no longer possesses the deleted knowledge:

```bash
# Cell 11: Query unlearned model for target entity
!python scripts/generate.py --shard-id 1 --unlearned --prompt "When was NeuroSync Diagnostics founded?"
```

```bash
# Cell 12: Query unlearned model for retained non-target entity
!python scripts/generate.py --shard-id 1 --unlearned --prompt "Where is Ashgrove Dynamics headquartered?"
```

---

## 8. Run Evaluation & Generate Erasure Report

Run all 6 probe suites (direct, paraphrased, reverse, multi-hop, non-target retention, and neighbor probes):

```bash
# Cell 13: Evaluate erasure efficacy
!python scripts/evaluate_sisa.py --target-group-id G001 --config configs/sisa_config.yaml
```

---

## 9. View Generated Erasure Report

Display the formatted Markdown report directly in Google Colab:

```python
# Cell 14: Render markdown erasure report in Colab
from IPython.display import display, Markdown

with open("outputs/reports/erasure_report.md", "r", encoding="utf-8") as f:
    report_md = f.read()

display(Markdown(report_md))
```
