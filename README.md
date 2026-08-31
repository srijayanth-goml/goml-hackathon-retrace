# ReTrace: SISA + LoRA Machine Unlearning for LLMs

A production-grade, modular implementation of **SISA (Sharded, Isolated, Sliced, and Aggregated)** machine unlearning with **PEFT LoRA** on `Qwen/Qwen2.5-1.5B-Instruct`.

---

## Table of Contents
1. [Core Concepts & Theoretical Framework](#core-concepts--theoretical-framework)
2. [ReTrace Architecture](#retrace-architecture)
3. [4-Shard & 4-Slice Partitioning Strategy](#4-shard--4-slice-partitioning-strategy)
4. [Incremental Slice Training Pipeline](#incremental-slice-training-pipeline)
5. [Target Unlearning & Rollback Mechanism](#target-unlearning--rollback-mechanism)
6. [Multi-Probe Verification & Metrics](#multi-probe-verification--metrics)
7. [LoRA Aggregation Trade-offs & Limitations](#lora-aggregation-trade-offs--limitations)
8. [Quickstart & CLI Manual](#quickstart--cli-manual)
9. [Google Colab Execution](#google-colab-execution)

---

## 1. Core Concepts & Theoretical Framework

Machine Unlearning aims to permanently excise the influence of specific training instances from a model upon request (e.g., GDPR "Right to be Forgotten", copyright removal, or factual erasure) without retraining the entire model from scratch.

### The SISA Paradigm (Bourtoule et al., 2021)
SISA achieves exact or certifiable unlearning through structured data isolation:
- **Sharded ($K$)**: The dataset $D$ is split into $K$ disjoint partitions: $D = \bigcup_{i=1}^K S_i$, with $S_i \cap S_j = \emptyset$.
- **Isolated**: Each shard $S_i$ trains its own parameter adapter $W_i$ independently.
- **Sliced ($R$)**: Each shard $S_i$ is subdivided into $R$ ordered slices: $S_i = (L_{i,1}, L_{i,2}, \dots, L_{i,R})$.
- **Aggregated**: At inference time, predictions are routed across shard adapters.

### Unlearning Time Complexity
When a deletion request arrives for an entity group residing in slice $k^*$ of shard $S^*$:
$$\text{Cost}_{\text{SISA}} = O\left(\frac{R - k^* + 1}{K \cdot R}\right) \times \text{Cost}_{\text{Full Retraining}}$$

For $K=4$ shards and $R=4$ slices ($16$ total slices across the system):
- Worst case ($k^* = 1$): Retrain 4 slices out of 16 $\implies$ **75.0% compute savings** (4x speedup).
- Best case ($k^* = 4$): Retrain 1 slice out of 16 $\implies$ **93.75% compute savings** (16x speedup).
- Average case: Retrain 2.5 slices $\implies$ **84.38% compute savings** (6.4x speedup).
- All other $K-1 = 3$ shards remain **100% untouched**.

---

## 2. ReTrace Architecture

```
                    Qwen-2.5-1.5B-Instruct (Frozen Base Model)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
      Shard 1 (S1)               Shard 2 (S2)               Shard 3 (S3)               Shard 4 (S4)
   ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
   │ L1: Groups G1-G7 │       │ L1: Groups G8-G14│       │ L1: Groups G15-21│       │ L1: Groups G22-28│
   │  └─> ckpt_S1_L1  │       │  └─> ckpt_S2_L1  │       │  └─> ckpt_S3_L1  │       │  └─> ckpt_S4_L1  │
   │ L2: Groups G29-34│       │ L2: Groups G35-40│       │ L2: Groups G41-46│       │ L2: Groups G47-52│
   │  └─> ckpt_S1_L2  │       │  └─> ckpt_S2_L2  │       │  └─> ckpt_S3_L2  │       │  └─> ckpt_S4_L2  │
   │ L3: Groups G53-58│       │ L3: Groups G59-64│       │ L3: Groups G65-70│       │ L3: Groups G71-76│
   │  └─> ckpt_S1_L3  │       │  └─> ckpt_S2_L3  │       │  └─> ckpt_S3_L3  │       │  └─> ckpt_S4_L3  │
   │ L4: Groups G77-82│       │ L4: Groups G83-88│       │ L4: Groups G89-94│       │ L4: Groups G95-100
   │  └─> ckpt_S1_L4  │       │  └─> ckpt_S2_L4  │       │  └─> ckpt_S3_L4  │       │  └─> ckpt_S4_L4  │
   └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
            │                          │                          │                          │
       Adapter S1                 Adapter S2                 Adapter S3                 Adapter S4
```

### LoRA Specifications
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (Parameters: 1.54B, completely frozen)
- **Rank ($r$)**: `16`
- **Scaling Factor ($\alpha$)**: `32`
- **Dropout**: `0.05`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters**: ~18.4M parameters per shard (< 1.2% of base model)

---

## 3. 4-Shard & 4-Slice Partitioning Strategy

The dataset (`knowledge_challenging_500 (1).xlsx`) consists of **500 core facts** grouped into **100 entity fact groups** (`fact_group_id`: `G001` - `G100`), spanning 53 companies and 47 high-profile professionals.

### Augmented Dataset (~2,264 Examples)
For each fact group, ReTrace synthesizes augmented probe and training instances across 5 key dimensions:
1. **Direct Fact Prompts**: Canonical factual statements and questions.
2. **Paraphrased Prompts**: Grammatical and stylistic permutations.
3. **Reverse Lookups**: Backward entity inference given attributes.
4. **Multi-Hop Probes**: Multi-step deductive queries chaining multiple attributes.
5. **Neighbor / Confusable Probes**: Attribute discrimination among similar entities.

### Strict Sharding Invariants
- **No Group Splitting**: All augmented instances belonging to `fact_group_id` $G$ are placed strictly within **one shard** and **one slice**.
- **Deterministic Partitioning**: Fixed random seed (`seed: 42`) guarantees exact reproducibility.
- **Partition Balance**: 25 entity groups per shard (~566 examples/shard), subdivided into 4 slices (~140 examples/slice).

---

## 4. Incremental Slice Training Pipeline

Training proceeds sequentially within each isolated shard:
1. **Slice 1 ($L_1$)**: Initializes fresh LoRA adapter from base model, trains on $L_1$, and checkpoints `ckpt_S{i}_L1`.
2. **Slice 2 ($L_2$)**: Loads `ckpt_S{i}_L1`, trains on $L_2$, and checkpoints `ckpt_S{i}_L2`.
3. **Slice 3 ($L_3$)**: Loads `ckpt_S{i}_L2`, trains on $L_3$, and checkpoints `ckpt_S{i}_L3`.
4. **Slice 4 ($L_4$)**: Loads `ckpt_S{i}_L3`, trains on $L_4$, and checkpoints `ckpt_S{i}_L4` (`final_adapter`).

---

## 5. Target Unlearning & Rollback Mechanism

When a deletion request is issued for `fact_group_id` (e.g. `G001`):

```
Step 1: Inspect Shards Metadata
        └── Target `G001` located in Shard 1, Slice 1.

Step 2: Rollback to Checkpoint
        └── Since slice == 1, roll back to base model (or ckpt_S1_L{k-1} if slice > 1).

Step 3: Filter Slice Data
        └── Remove all records with `fact_group_id == G001` from slice 1 data.

Step 4: Retrain Slices
        └── Train Slice 1 (filtered) -> ckpt_unlearned_S1_L1
        └── Train Slice 2 (original) -> ckpt_unlearned_S1_L2
        └── Train Slice 3 (original) -> ckpt_unlearned_S1_L3
        └── Train Slice 4 (original) -> ckpt_unlearned_S1_L4 (final unlearned adapter)

Step 5: Unaffected Shards
        └── Shards 2, 3, and 4 remain 100% untouched.
```

---

## 6. Multi-Probe Verification & Metrics

To verify true erasure and prevent silent memorization or unintended collateral degradation, ReTrace evaluates 6 probe suites:

| Metric | Formulation / Definition | Target Threshold |
| :--- | :--- | :--- |
| **Target Forgetting Rate** | $1 - \frac{\text{Correct Target Probes}}{\text{Total Target Probes}}$ | $\ge 95.0\%$ |
| **Target Leakage Rate** | Percentage of target entity outputs still recalled | $\le 5.0\%$ |
| **Non-Target Retention** | Accuracy on entities residing in other shards | $\ge 90.0\%$ |
| **Collateral Damage** | Accuracy drop on neighbor entities in same shard | $\le 5.0\%$ |
| **Speedup Factor** | $\frac{K \cdot R}{R - k^* + 1}$ vs full retraining baseline | $> 4.0\times$ |

---

## 7. LoRA Aggregation Trade-offs & Limitations

> [!CAUTION]
> **Why Simple LoRA Parameter Averaging Fails**:
> In linear classifiers, parameter averaging is mathematically sound. However, in deep transformer models with low-rank adapter factorizations ($\Delta W = B \cdot A$), naive parameter averaging:
> $$\overline{\Delta W} = \frac{1}{K} \sum_{i=1}^K B_i A_i \neq \left(\frac{1}{K}\sum B_i\right)\left(\frac{1}{K}\sum A_i\right)$$
> creates destructive interference in attention subspace projections and MLP representations, degrading model coherence.

### Recommended Aggregation Strategy:
1. **Isolated Shard Serving**: Keep shard adapters separate as modular expert adapters.
2. **Routing / Ensemble Selection**: Route queries to relevant shard adapter or evaluate through majority voting.

---

## 8. Quickstart & CLI Manual

### 1. Build Shards and Slices
```bash
python scripts/build_shards.py --config configs/sisa_config.yaml
```

### 2. Train SISA Shards
```bash
# Train all 4 shards (GPU / Colab)
python scripts/train_sisa.py --config configs/sisa_config.yaml

# Train a specific shard (e.g. Shard 1)
python scripts/train_sisa.py --shard-id 1

# Dry-run smoke test (Fast CPU verification)
python scripts/train_sisa.py --dry-run
```

### 3. Query Model Before Unlearning
```bash
python scripts/generate.py --shard-id 1 --prompt "When was NeuroSync Diagnostics founded?"
```

### 4. Unlearn an Entity Group
```bash
# Unlearn group G001
python scripts/unlearn_sisa.py --fact-group-id G001

# Dry-run unlearning
python scripts/unlearn_sisa.py --fact-group-id G001 --dry-run
```

### 5. Query Model After Unlearning
```bash
python scripts/generate.py --shard-id 1 --unlearned --prompt "When was NeuroSync Diagnostics founded?"
```

### 6. Comprehensive Evaluation & Erasure Report
```bash
python scripts/evaluate_sisa.py --target-group-id G001
```
Reports are automatically saved to:
- `outputs/reports/erasure_report.json`
- `outputs/reports/erasure_report.md`

---

## 9. Google Colab Execution

For a complete step-by-step notebook guide with zero setup friction on Free Google Colab T4 GPU, refer to [`colab_run.md`](file:///c:/Users/rithi/Documents/GoML/sisa/colab_run.md).
