"""
Module 4 (Verification & Erasure Report) config: signal thresholds, the general-
capability prompt set, the previous_company control-group check, and per-request
decoy-mention checks. Import from here instead of hardcoding across
direct_qa.py/mia.py/relational_probe.py/general_capability.py/report.py.

Reuses unlearning/config.py's drift tolerances rather than re-deciding them (Design
Doc Section 6's neighbor/general targets and this module's Impact Assessment section
should be judged against the SAME bar, not two independently-chosen ones) -- locked
recommendation, see plan.md's Module 4 Open Decisions.
"""
from __future__ import annotations

from pathlib import Path

import config as root_config
from unlearning import config as ul_config

VERIFICATION_DIR = Path(__file__).resolve().parent
REPORTS_DIR = VERIFICATION_DIR / "reports"

MODEL_NAME = root_config.MODEL_NAME

# --- Reused thresholds (locked decision: don't re-decide these here) -----------
NEIGHBOR_DRIFT_TOLERANCE = ul_config.NEIGHBOR_DRIFT_TOLERANCE
GENERAL_DRIFT_TOLERANCE = ul_config.GENERAL_DRIFT_TOLERANCE
FORGET_ACCURACY_COLLAPSE_THRESHOLD = ul_config.FORGET_ACCURACY_COLLAPSE_THRESHOLD

# --- MIA (mia.py) ---------------------------------------------------------------
# Locked recommendation (plan.md's Module 4 Open Decisions): percentile-rank against
# a heldout null, not a fitted likelihood-ratio test -- simple, and honest about what
# it can and can't claim on tiny forget sets.
MIA_NULL_SAMPLE_SIZE = 60             # how many heldout.jsonl forward-QA records form the null distribution
MIA_MIN_FORGET_SET_FOR_CONFIDENCE = 5  # below this, report.py must render the small-sample caveat inline
MIA_SEED = 271828                      # distinct from every other seed in the repo

# --- General capability spot-check (general_capability.py) ----------------------
# Locked recommendation: ~10-15 prompts, ALL mechanically gradable (single correct
# string) -- no open-ended prompts, so this module never needs an LLM-judge
# dependency, matching the rest of the repo's approach.
GENERAL_CAPABILITY_PROMPTS = [
    {"prompt": "What is 2 + 2? Answer with just the number.", "expected_substring": "4"},
    {"prompt": "What is 12 times 12? Answer with just the number.", "expected_substring": "144"},
    {"prompt": "What is the capital of France? Answer with just the city name.", "expected_substring": "paris"},
    {"prompt": "What is the capital of Japan? Answer with just the city name.", "expected_substring": "tokyo"},
    {"prompt": "How many days are there in a week? Answer with just the number.", "expected_substring": "7"},
    {"prompt": "What color do you get by mixing blue and yellow paint? Answer with just the color.", "expected_substring": "green"},
    {"prompt": "What is the chemical symbol for water? Answer with just the symbol.", "expected_substring": "h2o"},
    {"prompt": "What is the freezing point of water in Celsius? Answer with just the number.", "expected_substring": "0"},
    {"prompt": "What planet do we live on? Answer with just the planet name.", "expected_substring": "earth"},
    {"prompt": "What is the opposite of the word 'hot'? Answer with just the word.", "expected_substring": "cold"},
    {"prompt": "How many continents are there on Earth? Answer with just the number.", "expected_substring": "7"},
    {"prompt": "What is 100 divided by 4? Answer with just the number.", "expected_substring": "25"},
]

# --- Decoy-mention / over-forgetting checks (relational_probe.py) --------------
# Review doc's specific worked example (CLAUDE.md: "erase Silvergate Aerospace,
# verify the unrelated person whose employer is the decoy 'Silvergate Therapeutics'
# is untouched" -- corrected, per unlearning/requests/, to Silvergate LABS, since
# Silvergate Aerospace turned out to be a Module 1 heldout entity). Declared here as
# DATA, not hardcoded logic, so a new decoy check is a config entry, not new code --
# matches unlearning/requests/*.json's own "declare the demo case as data" pattern.
DECOY_CHECKS = [
    {
        "erased_entity": "Silvergate Labs",
        "decoy_value_substring": "Silvergate Therapeutics",
        "check_attribute": "current_company",
        "_comment": (
            "Silvergate Labs shares a name root with the decoy employer value "
            "'Silvergate Therapeutics' (someone's current_company -- not a real "
            "erasable company entity in this dataset). Verifies erasing Silvergate "
            "Labs does not collaterally damage the unrelated person whose employer "
            "happens to be the decoy."
        ),
    },
]
