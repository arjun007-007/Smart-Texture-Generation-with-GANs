# src/inference/generate_textures.py

import os
import argparse
import torch
from PIL import Image

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer_utils import load_checkpoint, move_to_device

def generate_samples(checkpoint_path, 
                     output_dir, 
                     num_samples=8, 
                     style_dim=512, 
                     image_size=256):
    """
    Generates random texture samples from a trained model.
    Args:
        checkpoint_path (str): Path to the trained model checkpoint (.pth file).
        output_dir (str): Where to save generated images.
        num_samples (int): Number of images to generate.
        style_dim (int): Dimension of style/latent vector.
        image_size (int): Desired output image size (the generator’s final output).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate generator and discriminator for checkpoint consistency
    generator = Generator(style_dim=style_dim, channels=[256, 128, 64, 32, 16])
    discriminator = Discriminator(channels=[16, 32, 64, 128, 256])

    device = move_to_device(generator, discriminator)

    # Load the trained weights (only need generator in practice, but we load both)
    _ = load_checkpoint(checkpoint_path, generator, discriminator)

    generator.eval()

    with torch.no_grad():
        for i in range(num_samples):
            style = torch.randn(1, style_dim, device=device)
            fake_image = generator(style)  # shape: (1, 3, H, W)

            # Convert to PIL image
            fake_image_cpu = fake_image.squeeze(0).detach().cpu()
            fake_image_cpu = (fake_image_cpu * 0.5 + 0.5).clamp_(0,1)  # if using normalize=True in transforms
            fake_image_pil = Image.fromarray((fake_image_cpu.permute(1,2,0).numpy() * 255).astype('uint8'))

            out_path = os.path.join(output_dir, f"generated_{i+1}.png")
            fake_image_pil.save(out_path)
            print(f"Saved generated texture at {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained checkpoint (.pth).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated samples.')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--style_dim', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        style_dim=args.style_dim,
        image_size=args.image_size
    )

if __name__ == "__main__":
    main()
