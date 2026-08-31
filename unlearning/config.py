"""
Module 3 (Unlearning Scripts) config: NPO/GA hyperparameters, batch composition,
early-stop thresholds, the forget-probe-split fraction, and the erasure-request
composition policy. Import from here instead of hardcoding across
selectors.py/data.py/npo.py/train.py.

Follows the repo convention (../CLAUDE.md): erasure/training parameters live in a
config file per module. See plan.md's "Module 3 -- Unlearning Scripts -- detailed
plan" for the reasoning behind each default, and ../CLAUDE.md for the locked-in
architecture decisions this must respect (NPO + neighbor-weighted retain as the
primary method, GA implemented only as a deliberately-worse comparison baseline,
runs locally against the Colab-trained baseline adapter).
"""
from __future__ import annotations

from pathlib import Path

import config as root_config          # repo-root config.py: paths, MODEL_NAME, RANDOM_SEED
from finetuning import config as ft_config   # reuse ONE checkpoints root (plan.md's Open Decisions)

UNLEARNING_DIR = Path(__file__).resolve().parent
REPORTS_DIR = UNLEARNING_DIR / "reports"
REQUESTS_DIR = UNLEARNING_DIR / "requests"

# --- Checkpoints: reuse Module 2's single checkpoints root + manifest.json (locked
# decision: "keep ONE checkpoints root rather than splitting revisions across two
# directories" -- the manifest is already the single source of truth for where a
# revision's adapter lives, so the physical layout follows it). ---
CHECKPOINTS_DIR = ft_config.CHECKPOINTS_DIR
MANIFEST_PATH = ft_config.CHECKPOINTS_DIR / "manifest.json"


def revision_checkpoint_dir(revision: int, method: str) -> Path:
    """Directory an unlearning run's adapter is saved to / loaded from, e.g.
    finetuning/checkpoints/revision-1-npo/."""
    return CHECKPOINTS_DIR / f"revision-{revision}-{method}"


# --- Erasure-request composition policy (locked decision: default to branching
# fresh from revision-0 for every request, to avoid the compounding-utility-loss
# risk flagged for sequential unlearning -- override with --parent-revision when a
# deliberately-sequential demo is wanted). ---
DEFAULT_PARENT_REVISION = 0

# --- Base model / adapter loading ---
MODEL_NAME = root_config.MODEL_NAME
BASELINE_ADAPTER_DIR = ft_config.BASELINE_CHECKPOINT_DIR  # revision-0

# --- NPO hyperparameters (Design Doc Section 6) -- starting points, not tuned; see
# plan.md's Open Decisions. ---
NPO_BETA = 0.1
LAMBDA_RETAIN = 1.0        # weight on the general-retain SFT loss term
LAMBDA_NEIGHBOR = 1.0      # reserved for a possible SEPARATE neighbor-specific loss term;
                           # the current implementation achieves neighbor emphasis by
                           # oversampling (see RETAIN_NEIGHBOR_PER_FORGET below) rather than a
                           # second weighted loss term -- kept here so that split is a config
                           # change, not a code change, if oversampling alone proves insufficient

# --- Gradient Ascent baseline-to-beat (Design Doc Section 6: "beta -> 0, no retain
# loss") -- deliberately conservative LR since GA has no NPO brake. ---
GA_LEARNING_RATE = 5e-5

# --- Batch composition: how many retain-general / retain-neighbor examples
# accompany each forget batch, and how much the neighbor pool is oversampled
# relative to the general-retain pool (Design Doc Section 6: "over-samples the
# confusable neighbors ... far more often than an unrelated ... company"). The
# oversampling comes from drawing NEIGHBOR_PER_FORGET items from a much SMALLER pool
# than GENERAL_PER_FORGET draws from -- same nominal batch share, much higher
# per-item selection frequency. ---
FORGET_BATCH_SIZE = 4
RETAIN_GENERAL_PER_FORGET = 4
RETAIN_NEIGHBOR_PER_FORGET = 4

# --- Training loop ---
LEARNING_RATE = 1e-4
MAX_STEPS = 300              # safety cap -- forget sets here are tiny (1 to 53 facts, Design Doc
                             # Section 3's table), so this should never actually bind
EVAL_EVERY_N_STEPS = 5
SEED = 424242                # distinct from Module 1's (42) and Module 2's (42 / 4242)
GENERAL_EVAL_SAMPLE_SIZE = 40  # subsample of retain_general used for the DURING-training
                               # accuracy check (the full pool is used for the final report)

# --- Early-stop-on-neighbor-drift rule (Design Doc Section 6: "stop as soon as the
# forget set has collapsed and BEFORE neighbor accuracy shows any drift"). ---
FORGET_ACCURACY_COLLAPSE_THRESHOLD = 0.1     # forget-set accuracy must fall to/below this
NEIGHBOR_DRIFT_TOLERANCE = 0.05               # neighbor accuracy may drop at most this many
                                              # percentage points from ITS OWN pre-unlearning value
GENERAL_DRIFT_TOLERANCE = 0.05                # same tolerance for the general-retain sample, as a
                                              # coarser wholesale-degradation guard

# --- Forget-probe split (this module's own gap-closer -- see plan.md's step 4: "a
# second gap in the same shape as the one Module 1 left for Module 2"). Named
# distinctly from every other split in the repo: Module 1's train/heldout,
# Module 2's sft-train/sft-val, and this module's forget-train/forget-probe are
# three different splits with three different purposes. ---
FORGET_PROBE_MIN_PER_FACT = 1       # hold back at least this many surface forms per targeted fact_id
FORGET_PROBE_FRACTION = 0.25          # ... or this fraction, whichever is larger, minus at least
                                     # 1 example always left in forget_train to actually train on
FORGET_PROBE_SEED = 131313            # distinct from every other split's seed in the repo


def training_args_as_dict(method: str) -> dict:
    return {
        "method": method,
        "learning_rate": LEARNING_RATE if method == "npo" else GA_LEARNING_RATE,
        "npo_beta": NPO_BETA if method == "npo" else None,
        "lambda_retain": LAMBDA_RETAIN if method == "npo" else None,
        "forget_batch_size": FORGET_BATCH_SIZE,
        "retain_general_per_forget": RETAIN_GENERAL_PER_FORGET,
        "retain_neighbor_per_forget": RETAIN_NEIGHBOR_PER_FORGET,
        "max_steps": MAX_STEPS,
        "eval_every_n_steps": EVAL_EVERY_N_STEPS,
        "seed": SEED,
    }
