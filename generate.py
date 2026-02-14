#!/usr/bin/env python3
"""
generate.py

Windows + PowerShell friendly CLI diffusion image generator.

Features:
- Text-to-image (SDXL)
- Optional img2img (content image)
- Style conditioning via IP-Adapter
- Multiple style images blended into ONE coherent style
- No UI
- input_images/  -> source images
- output_images/ -> generated results

IMPORTANT:
Uses SDXL-compatible IP-Adapter weight:
  ip-adapter_sdxl.safetensors
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# ---------------- CONFIG ----------------

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_SUBFOLDER = "sdxl_models"
IPADAPTER_WEIGHT = "ip-adapter_sdxl.safetensors"

# ----------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


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
    Blend multiple style images into ONE image.
    Prevents grid/panel artifacts and satisfies IP-Adapter's single-image rule.
    """
    if len(images) == 1:
        return images[0]

    resized = [im.resize((size, size), Image.BICUBIC) for im in images]
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in resized], axis=0)
    mean = arr.mean(axis=0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mean, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="SDXL + IP-Adapter image generator (CLI)")

    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")

    parser.add_argument("--input-dir", default="input_images", help="Input image directory")
    parser.add_argument("--output-dir", default="output_images", help="Output directory")

    parser.add_argument("--style-images", required=True,
                        help="Comma-separated style images (required)")
    parser.add_argument("--content-images", default="",
                        help="Comma-separated content images (optional)")

    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--style-scale", type=float, default=0.8)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # ---------- Load style images ----------
    style_files = csv_list(args.style_images)
    style_imgs = [load_rgb(input_dir / f) for f in style_files]
    style_image = blend_style_images(style_imgs)

    # ---------- Content images ----------
    content_files = csv_list(args.content_images)
    content_imgs = [load_rgb(input_dir / f) for f in content_files]
    use_img2img = len(content_imgs) > 0

    # ---------- Torch setup ----------
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Pipeline ----------
    if use_img2img:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            DEFAULT_MODEL, torch_dtype=torch_dtype
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            DEFAULT_MODEL, torch_dtype=torch_dtype
        )

    pipe.to(device)

    # ---------- IP-Adapter ----------
    pipe.load_ip_adapter(
        IPADAPTER_REPO,
        subfolder=IPADAPTER_SUBFOLDER,
        weight_name=IPADAPTER_WEIGHT,
    )
    pipe.set_ip_adapter_scale(args.style_scale)

    # ---------- Seed ----------
    generator = None
    if args.seed != -1:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    def generate(init_image: Optional[Image.Image], tag: str):
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
            name = timestamp_name(f"gen{tag}")
            img.save(output_dir / name)
            print("Saved:", output_dir / name)

    if use_img2img:
        for idx, img in enumerate(content_imgs):
            generate(img, f"_from_{idx}")
    else:
        generate(None, "")


if __name__ == "__main__":
    main()
