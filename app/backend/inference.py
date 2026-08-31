"""
Multi-turn chat generation for the live chat UI -- kept separate from
finetuning/eval_quick.py's generate_answer and unlearning/eval_during_unlearning.py's
accuracy_on, which are single-turn and exact-match-graded (scoring, not
conversation) and serve a different purpose. Heavy imports (torch) are lazy; call
only after the caller has already confirmed torch/transformers/peft are installed
(app.backend.adapters.AdapterCache does this via _require_heavy_deps() before ever
reaching here).
"""
from __future__ import annotations

from typing import List

from app.backend import config as be_config


def generate_chat_reply(model, tokenizer, messages: List[dict], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=be_config.DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
