import argparse
from pathlib import Path
from .styles import list_styles, load_style
from .generate import generate_one
from .prompt_txt_file_reader import read_prompts

# Main entry point for the command-line interface. 
# Parses arguments, loads the specified style, and calls the generation function. 
# The user can specify the prompt, style, model, generation parameters, and device options through command-line arguments. 
# The generated images will be saved to the output directory with information about the used style and generation settings.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="Single prompt string. Use the single prompt string or --prompts-file",)
    parser.add_argument("--style", required=True, help="Name of a folder under input_images/")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium") # Default to SD 3.5 Medium for better performance and compatibility with limited VRAM. This model can produce good quality images while being more accessible for users without high-end GPUs. 
    parser.add_argument("--steps", type=int, default=15) # Default to 15 steps for faster generation. This can be increased for higher quality at the cost of longer generation time, especially on CPU.
    parser.add_argument("--guidance", type=float, default=3.5) # Higher guidance scale encourages the model to follow the prompt more closely, while lower values allow for more creativity and variation. The default of 3.5 is a good starting point for balancing prompt adherence and creativity.
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512) # Default to 512x512 for better performance and compatibility with limited VRAM. 
                                                          #This model can generate larger images but will require more GPU memory and may be slower, especially on a CPU.

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda") # Use --device cpu to run on CPU (not recommended due to slowness, but can be used for testing or if no GPU is available). The script will automatically check if CUDA is available when "cuda" is specified and fall back to CPU if not.
    parser.add_argument("--cpu-offload", action="store_true") # Enable CPU offloading to reduce GPU memory usage. This will offload parts of the model to CPU when not in use, allowing it to run on GPUs with less VRAM. It may slow down generation but can help avoid out-of-memory errors on limited hardware.
    parser.add_argument("--no-t5", action="store_true") # Disable the T5 text encoder and use CLIP text encoder instead. This can reduce VRAM usage and speed up generation, but may result in less accurate style keyword extraction. It can be useful for testing or if the T5 model is causing memory issues.

    parser.add_argument("--prompts-file", type=str, default=None, help="Path to a .txt file containing multiple prompts.")  # Optional argument that specifies a text file that has multiple prompts. The prompts will be parsed according to the specified mode (line or blankline)
    parser.add_argument("--prompts-mode", type=str, default="blankline", choices=["line", "blankline"], help="How to parse prompts in the prompts file.") # If --prompts-file is provided, this option specifies how to parse the prompts from the file. "line" mode treats each non-empty line as a separate prompt, while "blankline" mode treats blocks of text separated by blank lines as individual prompts. The default is "blankline", which allows for multi-line prompts that can be more descriptive and detailed, while "line" mode is simpler and more concise for shorter prompts.
   
    #Example usage:
    #python -m src.main --prompt "A fantasy landscape with mountains and a river" --style "Van Gogh" --steps 20 --guidance 4.0 --device cuda --cpu-offload --no-t5

    args = parser.parse_args()

    # Validate that the user has provided either a single prompt or a prompts file, but not both. Also make sure that at least one of these options is provided. 
    if args.prompts_file and args.prompt:parser.error("Use either --prompt OR --prompts-file (not both).") # Make sure that the user only provides either a single prompt or a prompts file, but not both. 
    if not args.prompts_file and not args.prompt: parser.error("You must provide --prompt or --prompts-file.") # Make sure that the user provides at least one of the prompt options.

    # Load prompts from the specified file if provided, otherwise use the single prompt string. 
    if args.prompts_file:
        prompts = read_prompts(Path(args.prompts_file), mode=args.prompts_mode)
    else:
        prompts = [args.prompt]

    if not prompts:
        raise SystemExit("No prompts found. Check your prompt text or prompts file formatting.") 
    
    # Set up paths for styles and output. The styles are expected to be in the "input_images" folder, and the generated images will be saved to the "output_images" folder.
    project_root = Path(__file__).resolve().parent.parent
    styles_root = project_root / "input_images"
    out_root = project_root / "output_images"

    # Validate that the specified style exists in the styles directory. If not, print an error message with the list of available styles.
    available = list_styles(styles_root)
    if args.style not in available:
        raise SystemExit(f"Unknown style '{args.style}'. Available: {', '.join(available) if available else '(none found)'}")

    # Load the specified style.
    style = load_style(styles_root, args.style)

    # Print out some information about the style being used, including the number of files in the style folder.
    style_files = [p for p in style.folder.glob("*.*")]
    print("Using style:", style.name)
    print("Style folder:", style.folder.resolve())
    print("Files in style folder:", len(style_files))
    for p in style_files[:10]:
        print(" -", p.name)

    # Generate for each prompt
    all_paths = []
    total = len(prompts)

    # A progress line for batch runs 
    for idx, prompt in enumerate(prompts): print(f"\n[{idx + 1}/{total}] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    all_paths = []
    total = len(prompts)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{total}] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

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

        # generate_one returns list[Path]; handle either case safely
        if isinstance(paths, (list, tuple)):
            all_paths.extend(paths)
        else:
            all_paths.append(paths)

    print("Saved:")
    for p in all_paths:
        print(" -", p)

# Entry point for the script. When run directly, it will execute the main() function which handles the command-line interface and generation process.
if __name__ == "__main__":
    main()
