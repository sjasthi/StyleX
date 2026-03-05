from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Style embedding extraction using CLIP. Given a folder of reference images, it computes the average CLIP image embedding and finds the top-k closest text embeddings from a predefined style vocabulary
# The results include the matched keywords, their similarity scores, and the list of used reference images
# The style vocabulary is loaded from a text file, allowing for easy customization without code changes
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "style_vocab.txt"


def load_style_vocab(vocab_path: Path = DEFAULT_VOCAB_PATH) -> list[str]:
    
   # Load style vocabulary from a single .txt file (one phrase per line)
   # Ignore any blank lines
   # Ignore any lines that start with '#' to allow for comments in the vocab file
   # Strips whitespace
   # Deduplicates while preserving order 
    
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Style vocab file not found: {vocab_path}\n"
            f"Create it with one style phrase per line (e.g. 'anime style', 'realistic style', 'cyberpunk style')."
        )

    raw_lines = vocab_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen: set[str] = set()

    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out

# The results include the paths of the saved images
@dataclass
class StyleEmbeddingResult:
    keywords: list[str]
    scores: list[float]
    used_images: list[str]

def list_reference_images(style_folder: Path) -> list[Path]:
    if not style_folder.exists():
        return []
    paths = [p for p in style_folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


def _device_and_dtype(device: str) -> tuple[str, torch.dtype]:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    return ("cuda" if use_cuda else "cpu"), (torch.float16 if use_cuda else torch.float32)

# The results include the paths of the saved images
# Extracts style keywords from reference images using CLIP embeddings. Caches results in a JSON file within the style folder to speed up future runs with the same settings
# The style keywords are used to enhance the prompt for image generation, allowing the model to better capture the desired style based on the reference images
# The function handles loading and saving the cache, and ensures that the style keywords are only recomputed when necessary (e.g. if the top_k or model_id parameters change)
@torch.no_grad()
def extract_style_keywords(
    style_folder: Path,
    device: str = "cuda",
    top_k: int = 20,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    vocab_path: Path = DEFAULT_VOCAB_PATH,   # NEW
) -> StyleEmbeddingResult:
    img_paths = list_reference_images(style_folder)
    if not img_paths:
        return StyleEmbeddingResult(keywords=[], scores=[], used_images=[])

    # Load vocab from text file
    style_vocab = load_style_vocab(vocab_path)
    if not style_vocab:
        return StyleEmbeddingResult(keywords=[], scores=[], used_images=[p.name for p in img_paths])

    dev, _dtype = _device_and_dtype(device)

    model = CLIPModel.from_pretrained(clip_model_id)
    processor = CLIPProcessor.from_pretrained(clip_model_id)

    model = model.to(dev)
    if dev == "cuda":
        model = model.half()

    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    img_inputs = processor(images=imgs, return_tensors="pt")
    img_inputs = {k: v.to(dev) for k, v in img_inputs.items()}

    vision_out = model.vision_model(
        pixel_values=img_inputs["pixel_values"],
        return_dict=True,
    )
    pooled = vision_out.pooler_output
    image_features = model.visual_projection(pooled)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    style_vec = image_features.mean(dim=0, keepdim=True)
    style_vec = style_vec / style_vec.norm(dim=-1, keepdim=True)

    # Use style_vocab from .txt file
    # Compute text features for the style vocabulary and find the top-k closest keywords to the style vector
    # This allows us to extract relevant style keywords based on the reference images, which enhance the prompt for image generation
    text_inputs = processor(text=style_vocab, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(dev) for k, v in text_inputs.items()}

    text_out = model.text_model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs.get("attention_mask", None),
        return_dict=True,
    )
    text_pooled = text_out.pooler_output
    text_features = model.text_projection(text_pooled)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (text_features @ style_vec.T).squeeze(1)

    # Ensure top_k is within valid range
    top_k = max(0, min(int(top_k), sims.numel()))
    vals, idx = torch.topk(sims, k=top_k)

    # Map indices back to keywords and convert scores to floats for JSON serialization. Also include the list of used reference images in the results
    keywords = [style_vocab[i] for i in idx.tolist()] 
    scores = [float(v) for v in vals.tolist()]
    used = [p.name for p in img_paths]

    return StyleEmbeddingResult(keywords=keywords, scores=scores, used_images=used)