from __future__ import annotations
from pathlib import Path
from typing import List

def read_prompts(
    path: Path,
    mode: str = "blankline",   # a line or a blank line, or a hash
) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def is_hash_separator(line: str) -> bool:
        s = line.strip()
        return bool(s) and set(s) == {"#"}  # "#", "##", "###" etc.

    if any(is_hash_separator(l) for l in lines):
        blocks: list[str] = []
        current: list[str] = []
        for raw in lines:
            if is_hash_separator(raw):
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                continue
            if raw.strip() == "":
                continue
            current.append(raw.rstrip())
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]
    if any(l.strip() == "" for l in lines):
        blocks: list[str] = []
        current = []
        for raw in lines:
            if raw.strip() == "":
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(raw.rstrip())
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]
    prompts = [l.strip() for l in lines if l.strip() != ""]
    return prompts