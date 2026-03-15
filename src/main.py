import argparse
from pathlib import Path

from .generate import GenerateConfig, generate_one
from .prompt_txt_file_reader import read_prompts
from .styles import list_styles, load_style


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", default=None, help="Single prompt string. Use either --prompt or --prompts-file.") # The user can provide a single prompt directly via the command line using the --prompt argument, or they can specify a file containing multiple prompts using the --prompts-file argument 
    parser.add_argument("--style", required=True, help="Name of a folder under input_images/")  # The --style argument is required and should correspond to a folder name under the input_images directory. This folder will contain reference images that define the style for the generation process
    parser.add_argument("--model",default="black-forest-labs/FLUX.2-klein-4B", help="Model id on Hugging Face. Default is FLUX.2-klein-4B (Apache-2.0).",) # The --model argument allows the user to specify which FLUX.2 model to use for generation. The default is set to "black-forest-labs/FLUX.2-klein-4B", which is a specific FLUX.2 model available on Hugging Face
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.") # The --steps argument controls how many inference steps the FLUX.2 pipeline will run for each image generation. A higher number of steps can lead to better quality but will take more time, while a lower number of steps will be faster but may result in lower quality images
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale.") # The --guidance argument sets the guidance scale for the generation process, which controls how strongly the model follows the conditioning inputs (the user prompt and reference images). A higher guidance scale encourages the model to adhere more closely to the conditioning inputs, while a lower guidance scale allows for more creative freedom in the generated images
    parser.add_argument("--height", type=int, default=512) # The --height and --width arguments specify the dimensions of the generated images. The default is set to 512x512 pixels, which is a common size for image generation tasks. Users can adjust these values to generate larger or smaller images based on their needs and the capabilities of their hardware
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda") # The --device argument allows the user to specify whether to run the generation process on a CUDA-enabled GPU or on the CPU. By default, it is set to "cuda", which will use the GPU if available for faster generation. If the user does not have a compatible GPU or prefers to run on the CPU, they can set this argument to "cpu"

    parser.add_argument("--prompts-file", type=str, default=None, help="Path to a .txt file containing multiple prompts.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.") # The --seed argument allows the user to set a random seed for the generation process, which can help ensure reproducibility of results. If a seed is provided, the same prompts and styles will yield the same generated images across different runs. If no seed is provided, the generation process will be non-deterministic and may produce different results each time it is run

    parser.add_argument("--use-ref-images", action="store_true", help="Use images in the style folder as reference inputs (default).") # The --use-ref-images and --no-ref-images arguments control whether the images in the specified style folder should be used as reference inputs for the FLUX.2 generation process. By default, --use-ref-images is enabled, meaning that the images in the style folder will be used to condition the generation and help guide the output towards the desired style. If the user prefers to run a pure text-to-image generation without using any reference images from the style folder, they can use the --no-ref-images flag to disable this behavior
    parser.add_argument("--no-ref-images", action="store_true", help="Disable using style folder images; run pure txt2img.") # The --no-ref-images flag allows the user to disable the use of reference images from the style folder, effectively running a pure text-to-image generation process. This can be useful if the user wants to see how the model generates images based solely on the text prompt without any visual conditioning from the style images. 
    parser.add_argument("--ref-max-images", type=int, default=10, help="Max number of reference images to use from the style folder.")
    parser.add_argument("--ref-sample-mode", choices=["random", "first"], default="random")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload to save VRAM.") # The --cpu-offload flag enables CPU offloading, which can help save VRAM when running the generation process on a GPU. 

    args = parser.parse_args()

    if args.prompts_file and args.prompt:
        parser.error("Use either --prompt OR --prompts-file (not both).")
    if not args.prompts_file and not args.prompt:
        parser.error("You must provide --prompt or --prompts-file.")
    if args.use_ref_images and args.no_ref_images:
        parser.error("Use either --use-ref-images OR --no-ref-images (not both).")

    if args.prompts_file:
        prompts = read_prompts(Path(args.prompts_file))
    else:
        prompts = [args.prompt]

    if not prompts:
        raise SystemExit("No prompts found. Check your prompt text or prompts file formatting.")

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

    use_ref_images = not args.no_ref_images

    # Generate images for each prompt and collect the paths of the saved outputs. The code iterates through each prompt, prepares the generation configuration, and calls the generate_one function to perform the image generation. 
    # The paths of the generated images are collected in the all_paths list, which is printed at the end to show the user where the generated images have been saved. 
    # This allows the user to easily access and review the generated images based on their prompts and chosen style
    all_paths = []
    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{total}] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        cfg = GenerateConfig(
            model_id=args.model,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            device=args.device,
            cpu_offload=args.cpu_offload,
            use_ref_images=use_ref_images,
            ref_max_images=args.ref_max_images,
            ref_sample_mode=args.ref_sample_mode,
            seed=args.seed,
            num_outputs=1,
        )

        paths = generate_one(
            user_prompt=prompt,
            style=style,
            out_root=out_root,
            cfg=cfg,
        )

        all_paths.extend(paths)

    print("Saved:")
    for p in all_paths:
        print(" -", p)


if __name__ == "__main__":
    main()
