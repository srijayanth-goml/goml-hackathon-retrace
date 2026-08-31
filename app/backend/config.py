"""
Module 5 (App Backend) config: host/port, generation defaults, job-queue paths.
Import from here instead of hardcoding across adapters.py/inference.py/jobs.py/
routes/*.py -- same repo convention as every other module's config.py.

See plan.md's "Module 5 -- App Backend -- detailed plan" for the reasoning behind
each default, and CLAUDE.md for the locked-in architecture decisions this respects.
"""
from __future__ import annotations

from pathlib import Path

import config as root_config
from finetuning import ft_config

BACKEND_DIR = Path(__file__).resolve().parent
JOBS_DIR = BACKEND_DIR / "jobs"
JOBS_JSON_PATH = JOBS_DIR / "jobs.json"

# --- Model -----------------------------------------------------------------
MODEL_NAME = root_config.MODEL_NAME
BF16 = ft_config.BF16

# --- Server ------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000

# CORS: open to a local frontend dev server (Module 6 -- Vite's default port and a
# couple of common alternates). Judging runs entirely on one local machine, so this
# does not need to be more permissive than "localhost, a handful of dev ports".
CORS_ALLOW_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8080", "http://127.0.0.1:8080",
]

# --- Chat generation ---------------------------------------------------------
# Greedy decode by default (do_sample=False) -- determinism over variety, matching
# every other generation call in this repo (finetuning/eval_quick.generate_answer,
# unlearning/eval_during_unlearning): a judge asking the same question twice should
# get the same answer.
DO_SAMPLE = False
DEFAULT_MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_CAP = 256   # server-side cap regardless of what the client asks for --
                           # keeps one chat request from stalling the single-worker
                           # job queue behind it for too long

# --- Erasure-request submission ---------------------------------------------
AUTO_VERIFY_DEFAULT = True
DEFAULT_METHOD = "npo"

# --- Job history --------------------------------------------------------------
LOG_TAIL_MAX_LINES = 200   # last N stdout lines captured per job (unlearning.train.run
                           # and verification.run_verification.run both print progress)
