from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from .styles import Style
from .utils import timestamp

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# This file contains the core logic for generating images based on user prompts and styles. 
# It defines the GenerateConfig dataclass to hold configuration parameters for the generation process, and the generate_one function that performs the actual image generation using a specified model and style. 
# The code also includes helper functions to list images in a folder, load images using PIL, and pick reference images from a style folder based on specified criteria. 
# The generate_one function handles the setup of the generation pipeline, including loading the model, preparing reference images if needed, and saving the generated images to the output directory.
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
# It handles the entire generation pipeline, including loading the FLUX.2 model, preparing reference images.
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
    num_outputs: int = 1


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
# It prepares the FLUX.2 pipeline, selects reference images, and calls the pipeline to generate images based on the user prompt and reference images.
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

    time_id = timestamp()
    out_dir = out_root / style.name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, img in enumerate(out_images, start=1):
        out_path = out_dir / f"{time_id}_{i:02d}.png"
        img.save(out_path)
        paths.append(out_path)

    return paths
