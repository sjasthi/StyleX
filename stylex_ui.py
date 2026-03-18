import shutil
import subprocess
from pathlib import Path
import gradio as gr

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "input_images"
OUTPUT_DIR = PROJECT_DIR / "output_images"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_reference_images(style_name: str, files) -> int:
    # Saves uploaded images into input_images/<style>
    style_dir = INPUT_DIR / style_name.strip()
    style_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    if files:
        for file_obj in files:
            src = Path(file_obj)
            if src.exists():
                dest = style_dir / src.name
                shutil.copy2(src, dest)
                saved += 1

    return saved


def latest_image():
    # Finds newest generated image
    if not OUTPUT_DIR.exists():
        return None

    image_files = list(OUTPUT_DIR.rglob("*"))

    image_files = [
        p for p in image_files
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    if not image_files:
        return None

    image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(image_files[0])


def run_local(prompt, style_name, model_name, steps, guidance, ref_images):
    """
    LOCAL GENERATION ONLY

    NOTE:
    - Removed all Colab-related logic
    - Removed job.json
    - Removed Downloads-based loading
    """

    prompt = prompt.strip()
    style_name = style_name.strip()
    model_name = model_name.strip()

    if not prompt or not style_name or not model_name:
        return None, "Missing required fields."

    saved_count = save_reference_images(style_name, ref_images)

    cmd = [
        "python",
        "-m",
        "src.main",
        "--model",
        model_name,
        "--style",
        style_name,
        "--prompt",
        prompt,
        "--steps",
        str(steps),
        "--guidance",
        str(guidance),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    output_file = latest_image()

    log_text = (
        f"Saved reference images: {saved_count}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    if result.returncode != 0 or not output_file:
        return None, log_text + "\n\nNo image found."

    return output_file, log_text


# ===========================
# UI (CLEAN VERSION)
# ===========================

with gr.Blocks(title="StyleX UI") as demo:
    gr.Markdown("# StyleX UI")
    gr.Markdown("Local image generation only.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=5)
            style_name = gr.Textbox(label="Style", value="fantasy")
            model_name = gr.Textbox(label="Model", value="black-forest-labs/FLUX.2-klein-4B")
            steps = gr.Slider(5, 50, value=20, step=1)
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5)

            ref_images = gr.File(label="Reference Images", file_count="multiple")

            run_button = gr.Button("Generate")

        with gr.Column():
            output_image = gr.Image(label="Image", type="filepath")
            logs = gr.Textbox(label="Logs", lines=20)

    run_button.click(
        fn=run_local,
        inputs=[prompt, style_name, model_name, steps, guidance, ref_images],
        outputs=[output_image, logs],
    )


if __name__ == "__main__":
    demo.launch()