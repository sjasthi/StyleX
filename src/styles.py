from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Style:
    name: str
    folder: Path
    prompt_suffix: str = ""
    negative_prompt: str = ""

    # LoRA (Low-Rank Adaptation) used for fine-tuning models like Stable Diffusion. If lora_path is set, the LoRA weights will be loaded and applied with the given scale.
    lora_path: str = ""
    lora_scale: float = 0.8

    # Embedding-based style extraction will compute a style embedding from the reference images and compare it to a curated list of style descriptors (STYLE_VOCAB) using CLIP. 
    # The top-k most similar descriptors will be appended to the generation prompt.
    use_style_embeddings: bool = True
    embeddings_top_k: int = 25
    embeddings_model_id: str = "openai/clip-vit-base-patch32"


# Style descriptor vocabulary for embedding-based style extraction. Curated list of common style keywords.
def list_styles(styles_root: Path) -> list[str]:
    if not styles_root.exists():
        return []
    return sorted([p.name for p in styles_root.iterdir() if p.is_dir()])


# Load style configuration from a folder under input_images/. The folder may contain a style.json with additional settings.
# (Completely Optional) Example style.json:
# {
#   "prompt_suffix": "in the style of Van Gogh",
#   "negative_prompt": "blurry, low quality",
#   "lora_path": "lora_weights.safetensors",
#   "lora_scale": 0.8,
#   "use_style_embeddings": true,
#   "embeddings_top_k": 20,
#   "embeddings_model_id": "openai/clip-vit-base-patch32"
# }
def _read_style_json(folder: Path) -> dict:
    for fname in ("style.json", "styles.json"):
        p = folder / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


# Load a style by name (folder under input_images/) and read its configuration.
# Raises FileNotFoundError if the style folder does not exist.
# Returns a Style object with all the settings needed for generation.
# The style folder may contain reference images for embedding-based style extraction, as well as an optional style.json for additional settings(Could be used in final product).
def load_style(styles_root: Path, style_name: str) -> Style:
    folder = styles_root / style_name
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Style '{style_name}' not found in {styles_root}")

    data = _read_style_json(folder)

    return Style(
        name=style_name,
        folder=folder,
        prompt_suffix=data.get("prompt_suffix", ""),
        negative_prompt=data.get("negative_prompt", ""),
        lora_path=data.get("lora_path", ""),
        lora_scale=float(data.get("lora_scale", 0.8)),
        use_style_embeddings=bool(data.get("use_style_embeddings", True)),
        embeddings_top_k=int(data.get("embeddings_top_k", 6)),
        embeddings_model_id=data.get("embeddings_model_id", "openai/clip-vit-base-patch32"),
    )
