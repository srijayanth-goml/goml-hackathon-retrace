"""
Renders ChatExample records to/from the JSONL files Module 2 consumes. The chat
template itself (Qwen2.5-1.5B-Instruct's jinja template, and the assistant-only
loss mask over it) is applied at train time in Module 2 via the tokenizer -- this
module only needs to produce the {"messages": [...], "metadata": {...}} shape,
matching Design Doc Section 5's example record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from common.schema import ChatExample


def write_jsonl(examples: List[ChatExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
