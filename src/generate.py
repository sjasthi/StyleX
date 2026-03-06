from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from .ref_scoring import score_candidates_against_refs
from .styles import Style
from .utils import timestamp

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# This module implements the core generation logic, including loading the FLUX.2 pipeline, preparing reference images, generating candidate images, scoring and ranking them against the reference images, and saving the top-ranked outputs to disk. 
# The main function is 'generate_one', which takes a user prompt, a Style object, an output directory, and a GenerateConfig object with all the relevant settings for the generation process. 
# The function returns a list of Paths to the saved output images. 
# The code is structured to be modular and extensible, allowing for easy adjustments to the generation parameters and scoring mechanisms as needed.
def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def _load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

# This function is responsible for selecting reference images from the specified style folder to be used as conditioning inputs for the FLUX.2 generation process.
def _pick_reference_images(
    style_folder: Path,
    max_images: int,
    sample_mode: str = "random",
    generator: Optional[torch.Generator] = None,
) -> List[Image.Image]:
    paths = _list_images(style_folder)
    if not paths or max_images <= 0:
        return []

    max_images = min(max_images, len(paths))

    if sample_mode == "first":
        chosen = paths[:max_images]
    else:
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
        idx = torch.randperm(len(paths), generator=generator).tolist()
        chosen = [paths[i] for i in idx[:max_images]]

    return [_load_pil(p) for p in chosen]

# The generate_one function is the main entry point for generating images based on a user prompt and a specified style. 
# It handles the entire generation pipeline, including loading the FLUX.2 model, preparing reference images, generating candidate images, scoring and ranking them, and saving the top-ranked outputs to disk.
@dataclass
class GenerateConfig:
    model_id: str
    steps: int = 8
    guidance: float = 1.0
    height: int = 512
    width: int = 512
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    cpu_offload: bool = True
    use_ref_images: bool = True
    ref_max_images: int = 10
    ref_sample_mode: str = "random"
    seed: Optional[int] = None

    num_outputs: int = 4

    rank_outputs: bool = True
    keep_top_k: int = 1
    use_clip_score: bool = True
    use_siglip_score: bool = True
    use_dino_score: bool = True
    clip_weight: float = 0.4
    siglip_weight: float = 0.4
    dino_weight: float = 0.2
    score_dtype: torch.dtype = torch.float32

# The code includes helper functions to determine if a given model_id corresponds to a FLUX.2 model and to load the appropriate FLUX.2 pipeline based on the model_id.
def _is_flux2(model_id: str) -> bool:
    mid = model_id.lower()
    return "flux.2" in mid or "flux2" in mid

# This function loads the FLUX.2 pipeline based on the specified model_id and configuration settings. It checks if the model_id corresponds to a FLUX.2 model and loads the appropriate pipeline class (Flux2KleinPipeline or Flux2Pipeline). 
# It also handles moving the pipeline to the correct device (CPU or CUDA) and enabling CPU offload if specified in the configuration.
def _load_flux2_pipeline(model_id: str, cfg: GenerateConfig):
    from diffusers import Flux2KleinPipeline, Flux2Pipeline  

    if "klein" in model_id.lower():
        pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=cfg.torch_dtype)
    else:
        pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=cfg.torch_dtype)

    if cfg.device == "cuda" and torch.cuda.is_available():
        if cfg.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    return pipe

# The generate_one function controls the entire image generation process for a single user prompt and style. 
# It prepares the FLUX.2 pipeline, selects reference images, generates candidate images, scores and ranks them, and saves the top-ranked outputs to disk.
def generate_one(
    user_prompt: str,
    style: Style,
    out_root: Path,
    cfg: GenerateConfig,
) -> List[Path]:
    out_root.mkdir(parents=True, exist_ok=True)

    if not _is_flux2(cfg.model_id):
        raise ValueError(
            "This generate.py expects a FLUX.2 model_id (e.g., black-forest-labs/FLUX.2-klein-4B). "
            f"Got: {cfg.model_id}"
        )

    pipe = _load_flux2_pipeline(cfg.model_id, cfg)

# The function sets up the random seed for reproducibility if specified in the configuration. 
# It creates a torch.Generator for both the main generation process and the reference image selection process, ensuring that the same seed is used for both to maintain consistency in the results.
    if cfg.seed is None:
        generator = None
        ref_gen = torch.Generator(device="cpu").manual_seed(0)
    else:
        gen_device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed))
        ref_gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    ref_images: Optional[List[Image.Image]] = None
    if cfg.use_ref_images:
        ref_images = _pick_reference_images(
            style.folder,
            max_images=cfg.ref_max_images,
            sample_mode=cfg.ref_sample_mode,
            generator=ref_gen,
        )
        if ref_images:
            print(
                f"FLUX.2 reference images used: {len(ref_images)} "
                f"(mode={cfg.ref_sample_mode}, max={cfg.ref_max_images})"
            )
        else:
            print("FLUX.2 reference images: none found (running txt2img).")

    # Prepare the arguments for the FLUX.2 pipeline call, including the user prompt, generation parameters, and reference images if available.
    call_kwargs = dict(
        prompt=user_prompt,
        height=int(cfg.height),
        width=int(cfg.width),
        num_inference_steps=max(1, int(cfg.steps)),
        guidance_scale=float(cfg.guidance),
        generator=generator,
        num_images_per_prompt=max(1, int(cfg.num_outputs)),
    )
    if ref_images:
        call_kwargs["image"] = ref_images

    result = pipe(**call_kwargs)
    out_images: List[Image.Image] = list(result.images)

    # Initialize scored_outputs with None scores; these will be updated if ranking is performed, or remain None if ranking is skipped.
    scored_outputs: List[tuple[Image.Image, Optional[float]]] = [(img, None) for img in out_images]

    should_rank = (
        cfg.rank_outputs
        and ref_images is not None
        and len(ref_images) > 0
        and len(out_images) > 1
        and (cfg.use_clip_score or cfg.use_siglip_score or cfg.use_dino_score)
    )

    if should_rank:
        score_device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        scores = score_candidates_against_refs(
            ref_images=ref_images,
            cand_images=out_images,
            device=score_device,
            dtype=cfg.score_dtype,
            use_clip=cfg.use_clip_score,
            use_siglip=cfg.use_siglip_score,
            use_dino=cfg.use_dino_score,
            clip_weight=cfg.clip_weight,
            siglip_weight=cfg.siglip_weight,
            dino_weight=cfg.dino_weight,
        )

        scored_outputs = sorted(
            list(zip(out_images, scores)),
            key=lambda item: item[1],
            reverse=True,
        )

        keep_n = min(max(1, int(cfg.keep_top_k)), len(scored_outputs))
        scored_outputs = scored_outputs[:keep_n]

        print("Ranking enabled. Top scores:")
        for rank_idx, (_, score) in enumerate(scored_outputs, start=1):
            print(f"  #{rank_idx}: {score:.4f}")
    elif cfg.rank_outputs:
        print("Ranking skipped. Need reference images, more than one output, and at least one scorer enabled.")

    time_id = timestamp()
    out_dir = out_root / style.name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, (img, score) in enumerate(scored_outputs, start=1):
        if score is None:
            out_path = out_dir / f"{time_id}_{i:02d}.png"
        else:
            out_path = out_dir / f"{time_id}_{i:02d}_score_{score:.4f}.png"
        img.save(out_path)
        paths.append(out_path)

    return paths
