import argparse
import re
from pathlib import Path
from .styles import list_styles, load_style
from .generate import generate_one


def parse_prompt_file(path: Path) -> list[str]:
    """
    Reads prompt sections from a file formatted like:

    [1]
    prompt text...

    [2]
    prompt text...

    Returns a list of prompt strings (each section becomes one prompt).
    """
    text = path.read_text(encoding="utf-8")

    # Split on lines like [1], [2], [3] that appear on their own line
    parts = re.split(r"(?m)^\s*\[\d+\]\s*$", text)

    # Remove empty chunks and trim whitespace
    prompts = [p.strip() for p in parts if p.strip()]
    return prompts


# Main entry point for the command-line interface.
# Parses arguments, loads the specified style, and calls the generation function.
# The user can specify the prompt, style, model, generation parameters, and device options through command-line arguments.
# The generated images will be saved to the output directory with information about the used style and generation settings.
def main():
    parser = argparse.ArgumentParser()

    # Keep your original --prompt, but allow a prompt file for batch generation
    parser.add_argument("--prompt", help="Single prompt string")
    parser.add_argument("--prompt-file", help="Path to prompt.txt with [1],[2],... prompts")

    parser.add_argument("--style", required=True, help="Name of a folder under input_images/")
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-medium"
    )  # Default to SD 3.5 Medium for better performance and compatibility with limited VRAM.
    parser.add_argument(
        "--steps",
        type=int,
        default=15
    )  # Default to 15 steps for faster generation.
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5
    )  # Balance prompt adherence vs creativity.
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--width",
        type=int,
        default=512
    )  # Default to 512x512 for better performance and compatibility with limited VRAM.

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda"
    )  # Use --device cpu to run on CPU.
    parser.add_argument(
        "--cpu-offload",
        action="store_true"
    )  # Enable CPU offloading to reduce GPU memory usage.
    parser.add_argument(
        "--no-t5",
        action="store_true"
    )  # Disable the T5 text encoder.

    # Example usage:
    # python -m src.main --prompt "A fantasy landscape with mountains and a river" --style "Van Gogh" --steps 20 --guidance 4.0 --device cuda --cpu-offload --no-t5
    # python -m src.main --style noir --prompt-file prompt.txt --device cpu

    args = parser.parse_args()

    # Validate prompt input (must use exactly one)
    if not args.prompt and not args.prompt_file:
        raise SystemExit('You must provide --prompt "..." OR --prompt-file prompt.txt')
    if args.prompt and args.prompt_file:
        raise SystemExit("Use only one: --prompt OR --prompt-file (not both)")

    project_root = Path(__file__).resolve().parent.parent
    styles_root = project_root / "input_images"
    out_root = project_root / "output_images"

    available = list_styles(styles_root)
    if args.style not in available:
        raise SystemExit(
            f"Unknown style '{args.style}'. Available: {', '.join(available) if available else '(none found)'}"
        )

    style = load_style(styles_root, args.style)

    style_files = [p for p in style.folder.glob("*.*")]
    print("Using style:", style.name)
    print("Style folder:", style.folder.resolve())
    print("Files in style folder:", len(style_files))
    for p in style_files[:10]:
        print(" -", p.name)

    # Build prompt list (single or batch)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (project_root / prompt_path).resolve()

        prompts = parse_prompt_file(prompt_path)
        if not prompts:
            raise SystemExit(f"No prompts found in {prompt_path}")
    else:
        prompts = [args.prompt]

    print(f"\nGenerating {len(prompts)} prompt(s) with style '{args.style}'...")

    all_paths = []

    # Generate one image per prompt (using the selected style)
    for i, prompt in enumerate(prompts, start=1):
        print("\n" + "=" * 60)
        print(f"[Prompt {i}/{len(prompts)}]")
        print(prompt)
        print("=" * 60)

        paths = generate_one(
            model_id=args.model,
            user_prompt=prompt,
            style=style,
            out_root=out_root,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            device=args.device,
            cpu_offload=args.cpu_offload,
            no_t5=args.no_t5,
        )

        all_paths.extend(paths)

    print("Saved:")
    for p in all_paths:
        print(" -", p)


# Entry point for the script. When run directly, it will execute the main() function which handles the command-line interface and generation process.
if __name__ == "__main__":
    main()