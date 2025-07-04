import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

small_model = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

prompts = [
    "A small cat sitting on a chair in a park in a very colorful environment"
]
#    "A dreamlike landscape with floating islands and waterfalls under a starry sky.",
#    "A Roman soldier standing guard in front of the Colosseum during sunset.",
#    "A cyberpunk character with neon tattoos in a rain-soaked alley.
results = pipe(
    prompts,
    num_inference_steps=50,
    guidance_scale=3.5
)

images = results.images

# Save or display the images
for i, img in enumerate(images):
    img.save(f"image_{i}_{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png")  # Save each image
