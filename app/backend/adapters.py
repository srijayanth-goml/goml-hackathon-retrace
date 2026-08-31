"""
Loads the Qwen2.5-1.5B-Instruct base model ONCE and attaches every manifest
revision's LoRA adapter as a NAMED peft adapter on that ONE base model, switching
with model.set_adapter(...) before generating -- avoids reloading the full base
model on every revision switch, which matters a lot on the hardware this actually
runs on (see plan.md's Module 5 "hardware fact": a single local laptop, not a GPU
box, so a chat request and a training/verification job must never touch the model
at the same moment). Heavy imports (torch/transformers/peft) are lazy, same posture
as every _require_heavy_deps() elsewhere in this repo -- importing this module never
requires them installed; only actually loading the model does.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app.backend import config as be_config
from unlearning import manifest as ul_manifest

# Reentrant: AdapterCache's own public methods (generate/refresh) all take this,
# and jobs.py's worker also takes it for a training/verification run's ENTIRE
# duration (including the refresh() call it makes on success) -- RLock lets the
# same thread re-enter without deadlocking itself.
MODEL_LOCK = threading.RLock()


class HeavyDepsMissing(RuntimeError):
    """torch/transformers/peft aren't installed. Routes convert this to a 503, never
    a raw traceback -- the HTTP-shaped equivalent of every train.py/
    run_verification.py CLI's own _require_heavy_deps() SystemExit."""


def _require_heavy_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise HeavyDepsMissing(
            f"app/backend needs {missing} installed -- run `pip install -r requirements.txt` "
            f"from the repo root. This does NOT require a GPU (same posture as "
            f"unlearning/train.py and verification/run_verification.py)."
        )


class AdapterCache:
    """One base model, N named peft adapters. NOT thread-safe on its own -- callers
    (routes/chat.py via generate(), jobs.py via refresh()) rely on MODEL_LOCK for
    mutual exclusion; this class assumes the lock is already held by its public
    methods and never takes it internally, to avoid double-locking surprises."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._adapter_name_by_revision: Dict[int, str] = {}

    def _ensure_base_loaded(self) -> None:
        _require_heavy_deps()
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        manifest = ul_manifest.read_manifest()
        entries = {e["revision"]: e for e in manifest["revisions"]}
        if 0 not in entries:
            raise RuntimeError(
                "revision-0 not found in finetuning/checkpoints/manifest.json -- "
                "run Module 2's baseline fine-tune first."
            )
        baseline = entries[0]
        dtype = torch.bfloat16 if be_config.BF16 else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(be_config.MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(be_config.MODEL_NAME, torch_dtype=dtype)
        model = PeftModel.from_pretrained(
            base, baseline["adapter_path"], adapter_name="revision-0", is_trainable=False
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        self._tokenizer = tokenizer
        self._adapter_name_by_revision[0] = "revision-0"

    def refresh(self) -> List[int]:
        """Loads any manifest revision not yet attached as a named adapter. Returns
        the newly-loaded revisions. Call after a job completes so the new revision
        is immediately chattable -- no server restart. Caller must hold MODEL_LOCK."""
        self._ensure_base_loaded()
        manifest = ul_manifest.read_manifest()
        newly_loaded: List[int] = []
        for entry in manifest["revisions"]:
            rev = entry["revision"]
            if rev in self._adapter_name_by_revision:
                continue
            adapter_name = f"revision-{rev}"
            self._model.load_adapter(entry["adapter_path"], adapter_name=adapter_name)
            self._adapter_name_by_revision[rev] = adapter_name
            newly_loaded.append(rev)
        return newly_loaded

    def generate(self, revision: int, messages: List[dict], max_new_tokens: int) -> str:
        from app.backend.inference import generate_chat_reply

        with MODEL_LOCK:
            self._ensure_base_loaded()
            if revision not in self._adapter_name_by_revision:
                self.refresh()
            if revision not in self._adapter_name_by_revision:
                raise KeyError(f"revision {revision} not found in the manifest")
            self._model.set_adapter(self._adapter_name_by_revision[revision])
            return generate_chat_reply(self._model, self._tokenizer, messages, max_new_tokens)

    def adapter_label(self, revision: int) -> str:
        return self._adapter_name_by_revision.get(revision, f"revision-{revision} (not loaded)")


_cache: Optional[AdapterCache] = None


def get_cache() -> AdapterCache:
    global _cache
    if _cache is None:
        _cache = AdapterCache()
    return _cache
