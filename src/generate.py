import json
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

from .styles import Style
from .utils import timestamp
from .style_embed import extract_style_keywords

# Build the final prompt by combining the user prompt, and embedding-based style keywords 
# The style keywords are appended to the user prompt, separated by commas, to enhance the generation results with style-specific descriptors
def build_prompt(user_prompt: str, style_keywords: list[str]) -> str:
    parts = [user_prompt]
    if style_keywords:
        parts.append(", ".join(style_keywords))
    return ", ".join(parts)

# Main function to generate images using Stable Diffusion 3.5 Medium while extracting style keywords from reference images
# The function handles loading the model, applying the optional LoRA weights, building the prompt, and saving the generated images to the output directory
# The results include the paths of the saved images
# The function also includes optimizations for low VRAM systems, such as attention slicing and model CPU offloading, to allow larger models to run on limited hardware
from .style_embed import DEFAULT_VOCAB_PATH
vocab_mtime = DEFAULT_VOCAB_PATH.stat().st_mtime if DEFAULT_VOCAB_PATH.exists() else None

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
    steps: int = 30,
    guidance: float = 5.0,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
):

    # Generate using SD3.5 Medium while extracting style keywords from reference images.
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # Embedding-based style keywords (cached per style folder)
    style_keywords: list[str] = []
    cache_file = style.folder / ".style_keywords_cache.json"

    if style.use_style_embeddings:
        cache = _load_style_cache(cache_file)
        if cache and cache.get("top_k") == style.embeddings_top_k and cache.get("model_id") == style.embeddings_model_id and cache.get("vocab_mtime") == vocab_mtime:
            style_keywords = cache.get("keywords", [])
        else:
            # Extract style keywords from reference images using CLIP embeddings. Caches results in a JSON file within the style folder to speed up future runs with the same settings. 
            # The style keywords are used to enhance the prompt for image generation, allowing the model to better capture the desired style based on the reference images. 
            # The function handles loading and saving the cache, and ensures that the style keywords are only recomputed when necessary (e.g. if the top_k or model_id parameters change).
            from .style_embed import extract_style_keywords, DEFAULT_VOCAB_PATH
            emb = extract_style_keywords(
                style_folder=style.folder,
                device="cuda" if use_cuda else "cpu",
                top_k=style.embeddings_top_k,
                clip_model_id=style.embeddings_model_id,
                vocab_path=DEFAULT_VOCAB_PATH,
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
    pipe_kwargs["text_encoder_3"] = None
    pipe_kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **pipe_kwargs)

    # VRAM helpers (good for low VRAM systems)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # Model CPU offload allows running larger models on limited VRAM by keeping most of the model on CPU and only moving parts to GPU when needed 
    # This can significantly reduce VRAM usage at the cost of some speed, especially on CPU
    pipe.enable_model_cpu_offload()
    

    prompt = build_prompt(user_prompt, style_keywords)

    call_kwargs = dict(
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
    )

    # Generate images using the pipeline with the constructed prompt and specified parameters. The results include the generated images, which will be saved to the output directory
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
