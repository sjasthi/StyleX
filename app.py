import shutil
from pathlib import Path
from datetime import datetime

import spaces
import gradio as gr

from src.generate import GenerateConfig, generate_one
from src.styles import load_style

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_images"
OUTPUT_DIR = PROJECT_ROOT / "output_images"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_reference_images(style_name: str, files) -> Path:
    style_name = style_name.strip()
    if not style_name:
        raise ValueError("Style folder name is required.")

    style_dir = INPUT_DIR / style_name
    style_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files so each run uses only the newly uploaded references
    for item in style_dir.iterdir():
        if item.is_file():
            item.unlink()

    saved = 0
    for file_path in files or []:
        src = Path(file_path)
        if src.exists():
            shutil.copy2(src, style_dir / src.name)
            saved += 1

    if saved == 0:
        raise ValueError("Please upload at least one reference image.")

    return style_dir


@spaces.GPU(duration=180)
def run_generation(prompt, files, steps, guidance, width, height, seed):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")

    style_name = f"style_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_reference_images(style_name, files)
    style = load_style(INPUT_DIR, style_name)

    seed_value = None if seed in ("", None) else int(seed)

    cfg = GenerateConfig(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        steps=int(steps),
        guidance=float(guidance),
        width=int(width),
        height=int(height),
        device="cuda",
        cpu_offload=True,
        use_ref_images=True,
        ref_max_images=6,
        ref_sample_mode="random",
        seed=seed_value,
        num_outputs=1,
    )

    paths = generate_one(
        user_prompt=prompt.strip(),
        style=style,
        out_root=OUTPUT_DIR,
        cfg=cfg,
    )

    return str(paths[0])


with gr.Blocks(title="StyleX") as demo:
    gr.Markdown("# StyleX")
    gr.Markdown(
        "Generate a stylistic image from a text prompt and uploaded reference images."
    )

    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Prompt**")
                    gr.Markdown(
                        "Describe the image you want the model to create. "
                        "Be specific about the subject, setting, mood, or details you want included."
                    )
                    prompt = gr.Textbox(
                        lines=3,
                        placeholder="Example: A portrait of an Indian woman AI scientist working at a futuristic desk"
                    )

                    gr.Markdown("**Reference Images**")
                    gr.Markdown(
                        "Upload **3–10 images** that all share a similar visual style. "
                        "Around **6 images** is usually the sweet spot for best results."
                    )
                    files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Reference Images"
                    )

                    gr.Markdown("**Steps**")
                    gr.Markdown(
                        "Controls how many generation steps the model uses. "
                        "More steps can improve quality, but they also take longer."
                    )
                    steps = gr.Slider(1, 10, value=6, step=1, label="Steps")

                    gr.Markdown("**Guidance**")
                    gr.Markdown(
                        "Controls how strongly the model follows the prompt and style conditioning. "
                        "Higher values can make results more controlled, while lower values may be more flexible."
                    )
                    guidance = gr.Slider(1.0, 5.0, value=1.0, step=0.1, label="Guidance")

                    gr.Markdown("**Seed (optional)**")
                    gr.Markdown(
                        "Use a seed if you want more reproducible results. "
                        "Leave it blank for a random result each time."
                    )
                    seed = gr.Textbox(
                        label="Seed",
                        placeholder="Leave blank for random"
                    )

                    with gr.Row():
                        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        height = gr.Slider(256, 1024, value=512, step=64, label="Height")

                    run_btn = gr.Button("Generate Image")
                    output = gr.Image(label="Generated Image")

                    run_btn.click(
                        fn=run_generation,
                        inputs=[prompt, files, steps, guidance, width, height, seed],
                        outputs=output,
                    )

                with gr.Column(scale=1):
                    gr.Markdown(
                        """
## About the Project

**StyleX** is an AI-based image generation system that creates a new image from:
- a **text prompt**
- a set of **reference images** that represent a desired style

## How It Works
1. Enter a prompt describing the image you want.
2. Upload reference images with a similar visual style.
3. The system uses those images as style guidance.
4. A new image is generated that combines your prompt with the uploaded style references.

## Best Practices
- Use **3–10** reference images
- **6 images** usually works best
- Keep the reference images visually consistent
- Clear prompts often produce better results

## Notes
- Each run creates a temporary style folder automatically
- If no seed is provided, generation will be random
- Results may vary depending on prompt detail and reference image quality
- To get more runs in a single day, you will either need to pay for credits or pay for the Pro version of Hugging Face
                        """
                    )

        with gr.Tab("Project Info"):
            gr.Markdown(
                """
# StyleX Project Overview

**StyleX** is a AI image generation project focused on producing new images
from a user prompt while conditioning the output on user-provided reference images.

## Main Idea
Instead of generating an image only from text, StyleX also uses uploaded reference images
to guide the visual style of the output.

## Features
- Prompt-based image generation
- Multi-image style conditioning
- Support for uploaded style references
- Random or seeded generation
- A simple Gradio interface for interaction

## Recommended Usage
- Upload images that share a similar style
- Use a clear, descriptive prompt
- Try around 6 reference images for balanced results
- Lower the width and height to 512x512 for faster generation

## Purpose
The purpose of the project is to explore how AI systems can combine user intent
with visual style references to generate images in a consistent style.
                """
            )

if __name__ == "__main__":
    demo.launch()