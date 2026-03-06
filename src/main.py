import argparse
from pathlib import Path

from .generate import GenerateConfig, generate_one
from .prompt_txt_file_reader import read_prompts
from .styles import list_styles, load_style


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", default=None, help="Single prompt string. Use either --prompt or --prompts-file.") # The user can provide a single prompt directly via the command line using the --prompt argument, or they can specify a file containing multiple prompts using the --prompts-file argument 
    parser.add_argument("--style", required=True, help="Name of a folder under input_images/") # The --style argument is required and should correspond to a folder name under the input_images directory. This folder will contain reference images that define the style for the generation process
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B", help="Model id on Hugging Face. Default is FLUX.2-klein-4B (Apache-2.0).",# The --model argument allows the user to specify which FLUX.2 model to use for generation. The default is set to "black-forest-labs/FLUX.2-klein-4B", which is a specific FLUX.2 model available on Hugging Face
    )
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.") # The --steps argument controls how many inference steps the FLUX.2 pipeline will run for each image generation. A higher number of steps can lead to better quality but will take more time, while a lower number of steps will be faster but may result in lower quality images
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale.") # The --guidance argument sets the guidance scale for the generation process, which controls how strongly the model follows the conditioning inputs (the user prompt and reference images). A higher guidance scale encourages the model to adhere more closely to the conditioning inputs, while a lower guidance scale allows for more creative freedom in the generated images
    parser.add_argument("--height", type=int, default=512) # The --height and --width arguments specify the dimensions of the generated images. The default is set to 512x512 pixels, which is a common size for image generation tasks. Users can adjust these values to generate larger or smaller images based on their needs and the capabilities of their hardware
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda") # The --device argument allows the user to specify whether to run the generation process on a CUDA-enabled GPU or on the CPU. By default, it is set to "cuda", which will use the GPU if available for faster generation. If the user does not have a compatible GPU or prefers to run on the CPU, they can set this argument to "cpu"

    parser.add_argument("--prompts-file", type=str, default=None, help="Path to a .txt file containing multiple prompts.") #
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.") # The --seed argument allows the user to set a random seed for the generation process, which can help ensure reproducibility of results. If a seed is provided, the same prompts and styles will yield the same generated images across different runs. If no seed is provided, the generation process will be non-deterministic and may produce different results each time it is run

    parser.add_argument("--use-ref-images", action="store_true", help="Use images in the style folder as reference inputs (default).") # The --use-ref-images and --no-ref-images arguments control whether the images in the specified style folder should be used as reference inputs for the FLUX.2 generation process. By default, --use-ref-images is enabled, meaning that the images in the style folder will be used to condition the generation and help guide the output towards the desired style. If the user prefers to run a pure text-to-image generation without using any reference images from the style folder, they can use the --no-ref-images flag to disable this behavior
    parser.add_argument("--no-ref-images", action="store_true", help="Disable using style folder images; run pure txt2img.") # The --no-ref-images flag allows the user to disable the use of reference images from the style folder, effectively running a pure text-to-image generation process. This can be useful if the user wants to see how the model generates images based solely on the text prompt without any visual conditioning from the style images. 
    parser.add_argument("--ref-max-images", type=int, default=10, help="Max number of reference images to use from the style folder.")
    parser.add_argument("--ref-sample-mode", choices=["random", "first"], default="random")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload to save VRAM.") # The --cpu-offload flag enables CPU offloading, which can help save VRAM when running the generation process on a GPU. 

    parser.add_argument("--num-outputs", type=int, default=4, help="Number of candidate images to generate per prompt.") # The --num-outputs argument specifies how many candidate images the FLUX.2 pipeline should generate for each prompt. By default, it is set to 4, meaning that for each user prompt and style combination, the model will produce 4 different images. This allows the user to have multiple options to choose from and can be useful for finding the best match to the desired style and prompt
    parser.add_argument("--rank-outputs", action="store_true", default=True, help="Rank generated outputs against the reference images.") # The --rank-outputs flag enables the ranking of generated images against the reference images from the style folder. When this flag is set, the generated images will be scored based on their similarity to the reference images using metrics like CLIP, SigLIP, and DINO. The scores will then be used to rank the generated images, allowing the user to identify which outputs best match the desired style as defined by the reference images. If this flag is not set, the generated images will not be scored or ranked, and all outputs will be treated equally without any preference based on their similarity to the reference images
    parser.add_argument("--keep-top-k", type=int, default=1, help="How many top-ranked images to save.") # The --keep-top-k argument specifies how many of the top-ranked generated images to save to disk after the ranking process. By default, it is set to 1, meaning that only the single best-ranked image (the one that is most similar to the reference images based on the scoring metrics) will be saved for each prompt. If the user wants to save more than one of the top-ranked images, they can increase this value to keep the top 2, top 3, etc., allowing them to have multiple high-quality outputs to choose from based on their similarity to the desired style    
    parser.add_argument("--no-clip-score", action="store_true", help="Disable CLIP scoring.") # The --no-clip-score, --no-siglip-score, and --no-dino-score flags allow the user to disable specific scoring metrics used for ranking the generated images against the reference images. By default, all three scoring metrics (CLIP, SigLIP, and DINO) are enabled, meaning that the generated images will be evaluated based on their similarity to the reference images using these metrics. If the user prefers to exclude a particular metric from the scoring process, they can use the corresponding flag to disable it. For example, if the user does not want to use CLIP for scoring, they can set --no-clip-score to disable CLIP scoring while still using SigLIP and DINO for evaluation
    parser.add_argument("--no-siglip-score", action="store_true", help="Disable SigLIP scoring.")
    parser.add_argument("--no-dino-score", action="store_true", help="Disable DINO scoring.")
    parser.add_argument("--clip-weight", type=float, default=0.4, help="CLIP ensemble weight.")
    parser.add_argument("--siglip-weight", type=float, default=0.4, help="SigLIP ensemble weight.") # The --clip-weight, --siglip-weight, and --dino-weight arguments allow the user to specify the weights for each of the scoring metrics (CLIP, SigLIP, and DINO) when ranking the generated images against the reference images. These weights determine how much influence each metric has on the overall score used for ranking. By default, CLIP and SigLIP are given a weight of 0.4 each, while DINO is given a weight of 0.2. This means that CLIP and SigLIP will have a stronger influence on the ranking compared to DINO. If the user wants to adjust the importance of each metric in the scoring process, they can modify these weights accordingly to emphasize or de-emphasize certain metrics based on their preferences or the specific characteristics of their style and prompts
    parser.add_argument("--dino-weight", type=float, default=0.2, help="DINO ensemble weight.")

    args = parser.parse_args()

    if args.prompts_file and args.prompt:
        parser.error("Use either --prompt OR --prompts-file (not both).")
    if not args.prompts_file and not args.prompt:
        parser.error("You must provide --prompt or --prompts-file.")
    if args.use_ref_images and args.no_ref_images:
        parser.error("Use either --use-ref-images OR --no-ref-images (not both).")
    if args.num_outputs < 1:
        parser.error("--num-outputs must be at least 1.")
    if args.keep_top_k < 1:
        parser.error("--keep-top-k must be at least 1.")

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
            num_outputs=args.num_outputs,
            rank_outputs=args.rank_outputs,
            keep_top_k=args.keep_top_k,
            use_clip_score=not args.no_clip_score,
            use_siglip_score=not args.no_siglip_score,
            use_dino_score=not args.no_dino_score,
            clip_weight=args.clip_weight,
            siglip_weight=args.siglip_weight,
            dino_weight=args.dino_weight,
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
