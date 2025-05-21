# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "diffusers",
#     "torch",
#     "transformers",
#     "accelerate",
#     "click",
# ]
# ///

import click
import torch
from diffusers import DiffusionPipeline
from pathlib import Path

# # Check that MPS is available
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")

prompt_ideas = ["dog with fedora"]

@click.command()
@click.option("--model", default="model_of_choice", help="Model to use for image generation")
# @click.option("--prompt", required=True, help="Text prompt for the image")
@click.option("--negative-prompt", default="low quality, blurry, deformed", help="Negative prompt")
@click.option("--steps", default=30, type=int, help="Number of inference steps")
@click.option("--guidance-scale", default=10, type=float, help="Guidance scale")
def generate_image(model, negative_prompt, steps, guidance_scale): # add back prompt cli arg
    """Generate an image using Stable Diffusion on Mac (CPU)."""
    click.echo(f"Loading model: {model}")
    
    # Load the model for CPU use (no float16)
    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    
    # Set to CPU instead of CUDA
    pipe = pipe.to("cpu")
    
    # # Set to CPU instead of CUDA
    # pipe = pipe.to("mps")
    # # Recommended if your computer has < 64 GB of RAM
    # pipe.enable_attention_slicing()

    click.echo("Generating image with the following settings:")
    # click.echo(f"  Prompt: {prompt}")
    click.echo(f"  Negative prompt: {negative_prompt}")
    click.echo(f"  Steps: {steps}")
    click.echo(f"  Guidance scale: {guidance_scale}")

    for i, prompt in enumerate(prompt_ideas):

        print(prompt)
    
        # Create the image
        with torch.no_grad():
            click.echo("Starting generation (this may take a while on CPU)...")
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Save the image
        output_path = Path(f"test_{i}.png") # check back to output
        image.save(output_path)
        click.echo(f"Image saved to: {output_path.absolute()}")


if __name__ == "__main__":
    generate_image()
