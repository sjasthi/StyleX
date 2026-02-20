from __future__ import annotations
# Embedding-based "style extraction" from reference images 
# Main Idea:
# - Compute an image embedding for each reference image in input_images/<style>/
# - Average them -> style embedding
# - Compare to a curated list of style descriptors using text embeddings
# - Pick top-k descriptors and append them to the generation prompt
# Default backbone: CLIP ViT-B/32 (Can run on CPU or GPU).


from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Need to add error handling for not having supported image files in the style folder.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Style descriptor vocabulary for embedding-based style extraction. Curated list of common style keywords.
# (Words can be added as needed. The quality of the extracted keywords will depend on how well this vocabulary covers the style.)
STYLE_VOCAB: list[str] = [
    "anime style",
    "manga panel",
    "cel shading",
    "clean line art",
    "vibrant colors",
    "soft pastel colors",
    "high contrast lighting",
    "cinematic lighting",
    "dramatic shadows",
    "watercolor painting",
    "oil painting",
    "digital painting",
    "concept art",
    "3d render",
    "pixel art",
    "low poly 3d",
    "photorealistic",
    "film grain",
    "bokeh background",
    "neon cyberpunk",
    "futuristic cityscape",
    "retro synthwave",
    "comic book ink",
    "studio ghibli inspired",
    "minimalist illustration",
    "flat vector art",
    "highly detailed",
    "intricate details",
    "soft focus",
    "sharp focus",
    "dynamic composition",
    "symmetrical composition",
    "black and white",
    "monochrome",
    "manga style",
    "painting style",
    "drawing style",
    "sketch style",
    "fantasy art",
    "sci-fi art",
    "nature landscape",
    "urban cityscape",
    "soft lighting",
    "hard lighting",
    "volumetric lighting",
    "golden hour lighting",
    "moody lighting",
    "studio lighting",
    "rim lighting",
    "backlit subject",
    "foggy atmosphere",
    "misty environment",
    "dramatic sky",
    "overcast lighting",
    "professional photography",
    "portrait photography",
    "street photography",
    "cinematic photography",
    "depth of field",
    "shallow depth of field",
    "sharp focus",
    "ultra realistic",
    "high dynamic range",
    "hdr photography",
    "natural skin texture",
    "semi realistic painting",
    "stylized illustration",
    "fantasy illustration",
    "game concept art",
    "character design",
    "environment concept art",
    "matte painting",
    "detailed background",
    "hand painted texture",
    "realistic 3d render",
    "cgi rendering",
    "octane render style",
    "unreal engine style",
    "ray traced lighting",
    "global illumination",
    "subsurface scattering",
    "high detail textures",
    "gritty texture",
    "smooth surfaces",
    "weathered materials",
    "metallic reflections",
    "glass reflections",
    "centered composition",
    "rule of thirds composition",
    "high resolution",
    "extremely detailed",
    "masterpiece quality",
    "anime lighting",
    "dynamic pose",
    "expressive character",
    "clean coloring",
    "illustration style shading",
]


# Data class to hold the results of style embedding extraction, including the top-k keywords, their similarity scores, and the list of reference images used for the extraction.
@dataclass
class StyleEmbeddingResult:
    keywords: list[str]
    scores: list[float]
    used_images: list[str]

# List all image files in the style folder. These will be used for embedding-based style extraction.
def list_reference_images(style_folder: Path) -> list[Path]:
    if not style_folder.exists():
        return []
    paths = [p for p in style_folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)

# Extract style keywords from reference images using CLIP embeddings. Results are cached in a .style_keywords_cache.json file in the style folder to speed up subsequent runs with the same settings.
#CLIP is used to compute embeddings for the reference images and the style descriptors, and then the top-k most similar descriptors are returned as keywords to be appended to the generation prompt.
#CLIP uses embedding normalization and cosine similarity to compare the style embedding with the text embeddings of the descriptors. The device (CPU or GPU) can be specified, and the results are cached for efficiency.
#Embedding-based style extraction can help capture the visual characteristics of the reference images and translate them into descriptive keywords that enhance the generation prompt.
def _device_and_dtype(device: str) -> tuple[str, torch.dtype]:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    return ("cuda" if use_cuda else "cpu"), (torch.float16 if use_cuda else torch.float32)

# Main function to extract style keywords from reference images and return the top-k most similar descriptors from the STYLE_VOCAB.
# The results include the list of keywords, their similarity scores, and the names of the reference images used for the extraction.
@torch.no_grad()
def extract_style_keywords(
    style_folder: Path,
    device: str = "cuda",
    top_k: int = 20,
    clip_model_id: str = "openai/clip-vit-base-patch32",
) -> StyleEmbeddingResult:
    img_paths = list_reference_images(style_folder)
    if not img_paths:
        return StyleEmbeddingResult(keywords=[], scores=[], used_images=[])

    dev, _dtype = _device_and_dtype(device)

    model = CLIPModel.from_pretrained(clip_model_id)
    processor = CLIPProcessor.from_pretrained(clip_model_id)

    model = model.to(dev)
    if dev == "cuda":
        model = model.half()

    # Image embeddings are computed using the CLIP vision model and projection. The embeddings are normalized and averaged to create a single style embedding that represents the overall style of the reference images.
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    img_inputs = processor(images=imgs, return_tensors="pt")
    img_inputs = {k: v.to(dev) for k, v in img_inputs.items()}

    # Use the internal vision_model + projection so we always get a tensor
    # (CLIPProcessor can return PIL images or tensors depending on the input, but we want to ensure we have tensors for the model)
    # The vision_model processes the images and produces pooled features, which are then projected to the CLIP embedding space. 
    # The resulting image features are normalized and averaged to create the style embedding.
    vision_out = model.vision_model(
        pixel_values=img_inputs["pixel_values"],
        return_dict=True,
    )
    pooled = vision_out.pooler_output                      # (N, hidden)
    image_features = model.visual_projection(pooled)       # (N, D)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    style_vec = image_features.mean(dim=0, keepdim=True)
    style_vec = style_vec / style_vec.norm(dim=-1, keepdim=True)

    # Text embeddings are computed for the STYLE_VOCAB using the CLIP text model and projection. The text features are normalized, and cosine similarity is used to compare the style embedding with each text embedding. 
    # The top-k most similar descriptors are returned as keywords, along with their similarity scores and the names of the reference images used for the extraction.
    text_inputs = processor(text=STYLE_VOCAB, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(dev) for k, v in text_inputs.items()}

    text_out = model.text_model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs.get("attention_mask", None),
        return_dict=True,
    )
    text_pooled = text_out.pooler_output                   # (V, hidden) 
    text_features = model.text_projection(text_pooled)     # (V, D) 

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Similarity + top-k 
    sims = (text_features @ style_vec.T).squeeze(1)  # (V,)

    top_k = max(0, min(int(top_k), sims.numel()))
    vals, idx = torch.topk(sims, k=top_k)

    # Convert results to lists and return as a StyleEmbeddingResult dataclass instance. The keywords are retrieved from the STYLE_VOCAB using the indices of the top-k similar descriptors, and the similarity scores are converted to floats. 
    # The names of the reference images used for the extraction are also included in the result.
    keywords = [STYLE_VOCAB[i] for i in idx.tolist()]
    scores = [float(v) for v in vals.tolist()]
    used = [p.name for p in img_paths]

    return StyleEmbeddingResult(keywords=keywords, scores=scores, used_images=used)
