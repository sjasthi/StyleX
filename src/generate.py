import json
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

from .styles import Style
from .utils import timestamp
from .style_embed import extract_style_keywords

# Build the final prompt by combining the user prompt, style prompt suffix, and embedding-based style keywords. 
# The components are joined with commas, and any empty parts are filtered out to create a clean final prompt for generation.
def build_prompt(user_prompt: str, style: Style, style_keywords: list[str]) -> str:
    parts = [user_prompt]
    if style.prompt_suffix:
        parts.append(style.prompt_suffix)
    if style_keywords:
        parts.append(", ".join(style_keywords))
    return ", ".join([p for p in parts if p])

# Main function to generate images using Stable Diffusion 3.5 Medium while extracting style keywords from reference images. 
# The function handles loading the model, applying the optional LoRA weights, building the prompt, and saving the generated images to the output directory. 
# The results include the paths of the saved images.
def _load_style_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_style_cache(cache_path: Path, data: dict) -> None:
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate_one(
    model_id: str,
    user_prompt: str,
    style: Style,
    out_root: Path,
    steps: int = 20,
    guidance: float = 3.5,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
    cpu_offload: bool = True,
    no_t5: bool = True,
):

    # Generate using SD3.5 Medium while extracting style keywords from reference images.
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # Embedding-based style keywords (cached per style folder)
    style_keywords: list[str] = []
    cache_file = style.folder / ".style_keywords_cache.json"

    if style.use_style_embeddings:
        cache = _load_style_cache(cache_file)
        if cache and cache.get("top_k") == style.embeddings_top_k and cache.get("model_id") == style.embeddings_model_id:
            style_keywords = cache.get("keywords", [])
        else:
            emb = extract_style_keywords(
                style_folder=style.folder,
                device="cuda" if use_cuda else "cpu",
                top_k=style.embeddings_top_k,
                clip_model_id=style.embeddings_model_id,
            )
            style_keywords = emb.keywords
            _save_style_cache(cache_file, {
                "model_id": style.embeddings_model_id,
                "top_k": style.embeddings_top_k,
                "keywords": style_keywords,
                "scores": emb.scores,
                "used_images": emb.used_images,
            })

        print("Embedding style keywords:", style_keywords)

    # Load SD3 pipeline (with optional T5 disabled)
    pipe_kwargs = dict(torch_dtype=dtype)
    if no_t5:
        pipe_kwargs["text_encoder_3"] = None
        pipe_kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **pipe_kwargs)

    # VRAM helpers (good for low VRAM systems)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # Offload models to CPU when not in use (good for low VRAM systems). 
    # The UNet and text encoders will be moved to CPU when not actively used for generation, and moved back to GPU when needed. 
    # This can help fit larger models on limited VRAM, but may increase generation time due to the overhead of moving models between CPU and GPU.
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda" if use_cuda else "cpu")

    prompt = build_prompt(user_prompt, style, style_keywords)

    call_kwargs = dict(
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
    )

    # If the style has a negative prompt defined, it will be added to the generation call to help steer the output away from unwanted characteristics. 
    # The negative prompt can include descriptors that describe what should be avoided in the generated images, such as " blurry" or "low quality".
    negative_prompt = getattr(style, "negative_prompt", "") or ""
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    result = pipe(prompt, **call_kwargs)
    images = result.images

    # Save generated images to output directory with a timestamp and style name in the filename. The images will be saved in a subfolder named after the style under the main output_images directory.
    out_dir = out_root / style.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamp for the filename to ensure uniqueness and help organize outputs by generation time. The timestamp will be in the format YYYYMMDD_HHMMSS.
    ts = timestamp()
    out_paths = []
    for i, img in enumerate(images):
        out_path = out_dir / f"{ts}_{style.name}_{i}.png"
        img.save(out_path)
        out_paths.append(out_path)

    return out_paths
