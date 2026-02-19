import argparse
from pathlib import Path
from .styles import list_styles, load_style
from .generate import generate_one

# Main entry point for the command-line interface. 
# Parses arguments, loads the specified style, and calls the generation function. 
# The user can specify the prompt, style, model, generation parameters, and device options through command-line arguments. 
# The generated images will be saved to the output directory with information about the used style and generation settings.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--style", required=True, help="Name of a folder under input_images/")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--no-t5", action="store_true")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    styles_root = project_root / "input_images"
    out_root = project_root / "output_images"

    available = list_styles(styles_root)
    if args.style not in available:
        raise SystemExit(f"Unknown style '{args.style}'. Available: {', '.join(available) if available else '(none found)'}")

    style = load_style(styles_root, args.style)

    style_files = [p for p in style.folder.glob("*.*")]
    print("Using style:", style.name)
    print("Style folder:", style.folder.resolve())
    print("Files in style folder:", len(style_files))
    for p in style_files[:10]:
        print(" -", p.name)

    # Generate images using the specified model, user prompt, and style configuration. 
    # The generation process will include extracting style keywords from reference images if enabled, applying optional LoRA weights, and saving the generated images to the output directory. 
    # The paths of the saved images are printed at the end.
    paths = generate_one(
        model_id=args.model,
        user_prompt=args.prompt,
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

    print("Saved:")
    for p in paths:
        print(" -", p)

# Entry point for the script. When run directly, it will execute the main() function which handles the command-line interface and generation process.
if __name__ == "__main__":
    main()
