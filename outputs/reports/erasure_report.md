# SISA + LoRA Machine Unlearning Erasure Report

**Evaluation Timestamp**: `2026-08-30 11:18:05`  
**Target Entity**: `Cobalt Energy` (`G056`)  
**Assigned Shard & Slice**: Shard `1`, Slice `4`

---

## 1. Executive Summary Metrics

| Metric | Result | Target / Ideal | Status |
| :--- | :--- | :--- | :--- |
| **Target Forgetting Rate** | **100.00%** | 100.0% | [PASSED] |
| **Target Leakage Rate** | **0.00%** | 0.0% | [SAFE] |
| **Non-Target Retention** | **2.00%** | > 90.0% | [DEGRADED] |
| **Collateral Damage** | **0.00%** | < 5.0% | [MINIMAL] |
| **Theoretical Speedup** | **16.00x** | > 4.0x | SISA Isolation Benefit |

---

## 2. Multi-Probe Suite Breakdown

| Probe Suite | Test Cases | Accuracy (Before) | Accuracy (After) | Forgetting Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Direct Target Probes** | 15 | 0.0% | 0.0% | **100.0%** |
| **Paraphrased Probes** | 15 | 0.0% | 0.0% | **100.0%** |
| **Reverse Probes** | 5 | 0.0% | 0.0% | **100.0%** |
| **Multi-Hop Probes** | 3 | 0.0% | 0.0% | **100.0%** |
| **Neighbor / Confusable** | 50 | 12.0% | 12.0% | N/A (Retention) |
| **Non-Target Retention** | 50 | 2.0% | 2.0% | N/A (Retention) |

---

## 3. SISA Architecture Analysis

- **Isolation Guarantee**: Only Shard `1` was affected. Shards `[2, 3, 4]` were completely untouched and unmodified.
- **Rollback Efficiency**: Slices `1..3` within Shard `1` were preserved via checkpoint restoration.
- **Aggregation Note**: Shard adapters are stored and served as independent modular experts. Naive parameter averaging across LoRA weights is deliberately avoided to prevent catastrophic representation interference.
