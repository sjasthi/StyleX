from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Style:
    name: str
    folder: Path
    use_style_embeddings: bool = True
    embeddings_top_k: int = 15
    embeddings_model_id: str = "openai/clip-vit-base-patch32"


# Style descriptor vocabulary for embedding-based style extraction. Curated list of common style keywords.
def list_styles(styles_root: Path) -> list[str]:
    if not styles_root.exists():
        return []
    return sorted([p.name for p in styles_root.iterdir() if p.is_dir()])

# Load a style by name from the styles directory. Validates that the specified style exists and returns a Style object with the relevant information about the style folder and embedding settings
# The function checks if the style folder exists and is a directory, and raises an error if not. This ensures that the rest of the generation process has a valid style to work with
# The Style object returned by this function can then be used in the generation process to access the reference images and embedding settings for that style
def load_style(styles_root: Path, style_name: str) -> Style:
    folder = styles_root / style_name
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Style '{style_name}' not found in {styles_root}")
    return Style(
        name=style_name,
        folder=folder,
    )
