import torch
from datetime import datetime
from diffusers import  StableDiffusion3Pipeline

large_model = "stabilityai/stable-diffusion-3.5-large"

pipe = StableDiffusion3Pipeline.from_pretrained(large_model, torch_dtype=torch.bfloat16)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

prompt = "A small cat sitting on a chair in a park in a very colorful environment."

'''results = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    height=512,
    width=512
)'''

results = pipe(
    prompt,
    num_inference_steps=28,
    guidance_scale=3.5
)

images = results.images

# Save or display the images
for i, img in enumerate(images):
    img.save(f"image_{i}_{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png")  # Save each image
