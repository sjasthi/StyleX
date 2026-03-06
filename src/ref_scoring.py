from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import torch
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# This module provides functionality for scoring generated images against reference images using multiple vision encoders (CLIP, SigLIP, DINO).
# The main function `score_candidates_against_refs` computes a weighted similarity score for each candidate image based on its embeddings compared to the centroid of the reference image embeddings. 
# This can be used to rank or filter generated images based on how closely they match the style of the reference images. 
def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


@dataclass
class EnsembleModels:
    clip_model: object | None
    clip_processor: object | None
    siglip_model: object | None
    siglip_processor: object | None
    dino_model: object | None
    dino_processor: object | None
    device: str
    dtype: torch.dtype


_MODELS: dict[tuple, EnsembleModels] = {}

# This module is designed for evaluating the similarity of generated images to reference images using multiple vision encoders. 
# It includes functionality to load and cache models, compute embeddings, and calculate similarity scores.
def get_ensemble_models(
    *,
    device: str,
    dtype: torch.dtype,
    use_clip: bool,
    use_siglip: bool,
    use_dino: bool,
    clip_id: str = "openai/clip-vit-base-patch32", 
    siglip_id: str = "google/siglip-so400m-patch14-384",
    dino_id: str = "facebook/dinov2-base",
) -> EnsembleModels:
    
    # Lazy-load models and cache them by (device,dtype,model-ids,flags) so we don't reload if the same configuration is requested again
    key = (device, str(dtype), use_clip, use_siglip, use_dino, clip_id, siglip_id, dino_id)
    if key in _MODELS:
        return _MODELS[key]

    clip_model = clip_processor = None
    siglip_model = siglip_processor = None
    dino_model = dino_processor = None

    # We load each model only if it's enabled by the flags. This way we can save VRAM and loading time if the user only wants to use a subset of the encoders for scoring.
    if use_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_processor = CLIPProcessor.from_pretrained(clip_id)
        clip_model = CLIPModel.from_pretrained(clip_id)
        clip_model = clip_model.to(device)
        if device == "cuda":
            clip_model = clip_model.to(dtype)

        clip_model.eval()

    # SigLIP and DINO are optional additional encoders that can be used to provide a more robust similarity score. 
    # They may capture different aspects of the image style compared to CLIP, so using them in combination can give a better overall assessment of how closely the generated images match the reference style. 
    # However, they also require additional VRAM and loading time, so they are disabled by default and can be enabled with flags.
    if use_siglip:
        from transformers import SiglipVisionModel, SiglipImageProcessor
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_id)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_id)
        siglip_model = siglip_model.to(device)
        if device == "cuda":
            siglip_model = siglip_model.to(dtype)
        siglip_model.eval()

    if use_dino:
        from transformers import AutoImageProcessor, AutoModel
        dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        dino_model = AutoModel.from_pretrained(dino_id)
        dino_model = dino_model.to(device)
        if device == "cuda":
            dino_model = dino_model.to(dtype)
        dino_model.eval()

    em = EnsembleModels(
        clip_model=clip_model,
        clip_processor=clip_processor,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
        dino_model=dino_model,
        dino_processor=dino_processor,
        device=device,
        dtype=dtype,
    )
    _MODELS[key] = em
    return em


# This function computes the CLIP image embeddings for a list of PIL images. 
# It uses the CLIP processor to prepare the images and then passes them through the CLIP vision model to get the embeddings. 
# The resulting features are normalized to unit length.
@torch.no_grad()
def embed_clip_images(models: EnsembleModels, images: list[Image.Image]) -> torch.Tensor:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    assert models.clip_model is not None and models.clip_processor is not None
    proc = models.clip_processor
    model = models.clip_model
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
    pooled = vision_out.pooler_output
    feats = model.visual_projection(pooled)
    return _norm(feats.float())

# This function computes the SigLIP image embeddings for a list of PIL images. 
# It uses the SigLIP processor to prepare the images and then passes them through the SigLIP vision model to get the embeddings. 
# The resulting features are normalized to unit length.
@torch.no_grad()
def embed_siglip_images(models: EnsembleModels, images: list[Image.Image]) -> torch.Tensor:
    assert models.siglip_model is not None and models.siglip_processor is not None
    proc = models.siglip_processor
    model = models.siglip_model
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    out = model(pixel_values=inputs["pixel_values"], return_dict=True)
    # SigLIP VisionModel exposes pooler_output
    feats = out.pooler_output
    return _norm(feats.float())

# This function computes the DINO image embeddings for a list of PIL images. 
# It uses the DINO processor to prepare the images and then passes them through the DINO
@torch.no_grad()
def embed_dino_images(models: EnsembleModels, images: list[Image.Image]) -> torch.Tensor:
    assert models.dino_model is not None and models.dino_processor is not None
    proc = models.dino_processor
    model = models.dino_model
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    out = model(**inputs, return_dict=True)
    # DINOv2: use mean pooled last_hidden_state as a global descriptor
    hidden = out.last_hidden_state
    feats = hidden.mean(dim=1)
    return _norm(feats.float())

# This function computes a similarity score for each candidate image based on its embeddings compared to the centroid of the reference image embeddings. 
# The score is a weighted sum of cosine similarities from multiple encoders (CLIP, SigLIP, DINO), and the weights can be adjusted by the user.
@torch.no_grad()
def score_candidates_against_refs(
    *,
    ref_images: list[Image.Image],
    cand_images: list[Image.Image],
    device: str,
    dtype: torch.dtype,
    use_clip: bool = True,
    use_siglip: bool = True,
    use_dino: bool = True,
    clip_weight: float = 0.4,
    siglip_weight: float = 0.4,
    dino_weight: float = 0.2,
    clip_id: str = "openai/clip-vit-base-patch32",
    siglip_id: str = "google/siglip-so400m-patch14-384",
    dino_id: str = "facebook/dinov2-base",
) -> list[float]:

    #Returns a list of scores (higher is better), one per candidate image.
    #Score is a weighted cosine similarity between candidate embeddings and the reference centroid.
    # Normalize weights
    weights = []
    if use_clip:
        weights.append(max(0.0, float(clip_weight)))
    else:
        clip_weight = 0.0
    if use_siglip:
        weights.append(max(0.0, float(siglip_weight)))
    else:
        siglip_weight = 0.0
    if use_dino:
        weights.append(max(0.0, float(dino_weight)))
    else:
        dino_weight = 0.0
    s = clip_weight + siglip_weight + dino_weight
    if s <= 0:
        # Nothing enabled -> all equal
        return [0.0 for _ in cand_images]
    clip_weight /= s
    siglip_weight /= s
    dino_weight /= s

    models = get_ensemble_models(
        device=device,
        dtype=dtype,
        use_clip=use_clip,
        use_siglip=use_siglip,
        use_dino=use_dino,
        clip_id=clip_id,
        siglip_id=siglip_id,
        dino_id=dino_id,
    )

    scores = torch.zeros(len(cand_images), device="cpu", dtype=torch.float32)

    # Batch in small chunks to be kind to VRAM
    def batches(xs: list[Image.Image], bs: int = 4) -> Iterable[list[Image.Image]]:
        for i in range(0, len(xs), bs):
            yield xs[i : i + bs]

    # Precompute ref centroids once per encoder
    ref_centroids: dict[str, torch.Tensor] = {}

    if use_clip:
        ref_emb = torch.cat([embed_clip_images(models, b) for b in batches(ref_images)], dim=0)
        ref_centroids["clip"] = _norm(ref_emb.mean(dim=0, keepdim=True))
    if use_siglip:
        ref_emb = torch.cat([embed_siglip_images(models, b) for b in batches(ref_images)], dim=0)
        ref_centroids["siglip"] = _norm(ref_emb.mean(dim=0, keepdim=True))
    if use_dino:
        ref_emb = torch.cat([embed_dino_images(models, b) for b in batches(ref_images)], dim=0)
        ref_centroids["dino"] = _norm(ref_emb.mean(dim=0, keepdim=True))

    # Candidate embeddings and similarities
    # We compute the embedding for each candidate image and then take the dot product with the reference centroid for each enabled encoder, weighted by the user-specified weights. 
    # The final score is a single float per candidate image that represents how closely it matches the style of the reference images according to the selected encoders.
    cand_scores = torch.zeros(len(cand_images), dtype=torch.float32)

    # Keep a running index so batching is easy
    idx0 = 0
    for b in batches(cand_images):
        n = len(b)
        part = torch.zeros(n, dtype=torch.float32)
        if use_clip:
            ce = embed_clip_images(models, b).cpu()
            part += clip_weight * (ce @ ref_centroids["clip"].cpu().T).squeeze(1)
        if use_siglip:
            ce = embed_siglip_images(models, b).cpu()
            part += siglip_weight * (ce @ ref_centroids["siglip"].cpu().T).squeeze(1)
        if use_dino:
            ce = embed_dino_images(models, b).cpu()
            part += dino_weight * (ce @ ref_centroids["dino"].cpu().T).squeeze(1)
        cand_scores[idx0:idx0+n] = part
        idx0 += n

    return [float(x) for x in cand_scores.tolist()]
