import torch
from pathlib import Path
import time  
from diffusers import  StableDiffusion3Pipeline

medium_model = "stabilityai/stable-diffusion-3.5-medium" # The latest and best model for image generation, but it is much slower and requires a powerful GPU to run. It can't run on a CPU. You can change the end "3.5-large" to "3.5-small" to use a smaller version of the model that can run on a CPU, but it will have worse image generation quality than the large model.

pipe = StableDiffusion3Pipeline.from_pretrained(medium_model, torch_dtype=torch.bfloat16)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda") # Leave as cuda if you have a NVIDIA GPU.

prompt = "a cyperpunk landscape"

results = pipe(
    prompt,
    num_inference_steps=15, # The number of steps for the diffusion process. More steps usually lead to higher quality images but take more time.
    guidance_scale=3.5,
    height=512,
    width=512
)

images = results.images

output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")

# Save or display the images
for i, img in enumerate(images):
    img.save(output_dir / f"{timestamp}_image_{i}.png")  # Save each image

print("Image/s saved to output_images")