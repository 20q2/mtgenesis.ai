import argparse
from diffusers import AutoPipelineForText2Image
import torch
import os

def main(prompt: str, width: int = 384, height: int = 288, output_path: str = None):
    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        variant = "fp16"
    else:
        device = "cpu"
        torch_dtype = torch.float32
        variant = None
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch_dtype,
        variant=variant
    )
    pipe.to(device)

    image = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        width=width,
        height=height
    ).images[0]

    if output_path:
        # Save the image to the specified path
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    else:
        # Show the image if no output path specified
        image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using SDXL-Turbo.")
    parser.add_argument("prompt", type=str, help="The prompt to generate an image from.")
    parser.add_argument("--width", type=int, default=384, help="Width of the generated image (default: 384)")
    parser.add_argument("--height", type=int, default=288, help="Height of the generated image (default: 288)")
    parser.add_argument("--output", type=str, help="Path to save the generated image")
    args = parser.parse_args()

    main(args.prompt, args.width, args.height, args.output)

