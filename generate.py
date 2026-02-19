#!/usr/bin/env python3
"""
generate.py

Windows + VS Code + PowerShell friendly SDXL image generator.

Supports:
- Text-to-image
- Multiple style images blended into ONE style reference
- No UI

Style folders live in:
  input_images/styles/<style-set>/
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# ---------------- CONFIG ----------------

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

DEFAULT_INPUT_DIR = "input_images"
DEFAULT_OUTPUT_DIR = "output_images"

STYLES_ROOT = "styles"  # subfolder under input_images/

IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_SUBFOLDER = "sdxl_models"
IPADAPTER_WEIGHT = "ip-adapter_sdxl.safetensors"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ---------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def timestamp_name(prefix: str, ext: str = "png") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{int(time.time() * 1000) % 1000000:06d}.{ext}"


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def blend_style_images(images: List[Image.Image], size: int = 512) -> Image.Image:
    """
    Blend multiple style images into ONE image to avoid panel artifacts.
    """
    if not images:
        raise ValueError("No style images provided.")

    if len(images) == 1:
        return images[0]

    resized = [im.resize((size, size), Image.BICUBIC) for im in images]
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in resized], axis=0)
    mean = arr.mean(axis=0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mean, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="SDXL + IP-Adapter image generator (style folders)")

    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")

    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # Option A: style folders
    parser.add_argument("--style-set", required=True,
                        help="Name of style folder under input_images/styles/")
    parser.add_argument("--style-glob", default="*",
                        help="Glob pattern for style images (default: all)")

    # Optional content images (img2img)
    parser.add_argument("--content-images", default="",
                        help="Comma-separated content images in input_images/content/")

    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--style-scale", type=float, default=0.8)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # ---------------- Load style set ----------------
    style_dir = input_dir / STYLES_ROOT / args.style_set
    if not style_dir.exists():
        raise SystemExit(f"ERROR: style-set not found: {style_dir}")

    style_imgs: List[Image.Image] = []
    for p in sorted(style_dir.glob(args.style_glob)):
        if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
            style_imgs.append(load_rgb(p))

    if not style_imgs:
        raise SystemExit(f"ERROR: no images found in style-set: {style_dir}")

    style_image = blend_style_images(style_imgs)

    # ---------------- Content images ----------------
    content_imgs: List[Image.Image] = []
    if args.content_images:
        content_dir = input_dir / "content"
        for name in args.content_images.split(","):
            p = content_dir / name.strip()
            if not p.exists():
                raise SystemExit(f"ERROR: content image not found: {p}")
            content_imgs.append(load_rgb(p))

    use_img2img = len(content_imgs) > 0

    # ---------------- Torch setup ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.float16 if args.dtype == "fp16"
        else torch.bfloat16 if args.dtype == "bf16"
        else torch.float32
    )

    # ---------------- Pipeline ----------------
    if use_img2img:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            DEFAULT_MODEL, torch_dtype=torch_dtype
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            DEFAULT_MODEL, torch_dtype=torch_dtype
        )

    pipe.to(device)

    pipe.load_ip_adapter(
        IPADAPTER_REPO,
        subfolder=IPADAPTER_SUBFOLDER,
        weight_name=IPADAPTER_WEIGHT,
    )
    pipe.set_ip_adapter_scale(args.style_scale)

    generator = None
    if args.seed != -1:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # ---------------- Generate ----------------
    def run(init_image: Optional[Image.Image], tag: str):
        for _ in range(args.num_images):
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt or None,
                ip_adapter_image=style_image,
                image=init_image if use_img2img else None,
                strength=args.strength if use_img2img else None,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            )
            img = result.images[0]
            out = output_dir / timestamp_name(f"gen{tag}")
            img.save(out)
            print("Saved:", out)

    if use_img2img:
        for i, img in enumerate(content_imgs):
            run(img, f"_from_{i}")
    else:
        run(None, "")


if __name__ == "__main__":
    main()
