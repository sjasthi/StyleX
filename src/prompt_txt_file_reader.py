from __future__ import annotations
from pathlib import Path
from typing import List

def read_prompts(
    path: Path,
    mode: str = "blankline",   # a line or a blank line
) -> List[str]:
    text = path.read_text(encoding="utf-8")

    if mode == "line":
        prompts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
        return prompts

    if mode == "blankline":
        blocks = []
        current: list[str] = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if line.strip().startswith("#"):
                continue
            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]

    raise ValueError("mode must be 'line' or 'blankline'")