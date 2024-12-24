# src/training/train.py

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Import your project modules
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.dataset import TextureDataset
from src.training.trainer_utils import (
    get_adversarial_loss_fn, build_optimizers, save_checkpoint, load_checkpoint, move_to_device
)

def train_gan(
    data_root,
    output_dir,
    epochs=50,
    batch_size=16,
    lr=0.0002,
    betas=(0.5, 0.999),
    image_size=256,
    checkpoint_path=None,
    save_interval=5
):
    """
    Main training function for the GAN.
    Args:
        data_root (str): Path to the processed dataset of images.
        output_dir (str): Directory to save checkpoints and logs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate for both G and D.
        betas (tuple): Betas for Adam optimizers.
        image_size (int): Square size for images.
        checkpoint_path (str): Path to a checkpoint to resume training.
        save_interval (int): Save a checkpoint every N epochs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create the dataset and dataloader
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    dataset = TextureDataset(root_dir=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 2. Initialize models
    generator = Generator(style_dim=512, channels=[256, 128, 64, 32, 16])
    discriminator = Discriminator(channels=[16, 32, 64, 128, 256])

    # 3. Move to GPU if available
    device = move_to_device(generator, discriminator)

    # 4. Build optimizers and loss functions
    g_optimizer, d_optimizer = build_optimizers(generator, discriminator, lr=lr, betas=betas)
    d_loss_fn, g_loss_fn = get_adversarial_loss_fn()

    start_epoch = 0
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)

    # 5. Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        g_losses = []
        d_losses = []

        for batch_idx, real_images in enumerate(dataloader):
            # Move data to GPU (if available)
            real_images = real_images.to(device)

            # ========== Train Discriminator ==========
            discriminator.zero_grad(set_to_none=True)

            # Generate latent (z) and style input
            batch_size_curr = real_images.size(0)
            style = torch.randn(batch_size_curr, generator.style_dim, device=device)
            fake_images = generator(style)

            # Forward pass
            real_pred = discriminator(real_images)
            fake_pred = discriminator(fake_images.detach())

            # Compute D loss
            d_loss = d_loss_fn(real_pred.view(-1), fake_pred.view(-1))
            d_loss.backward()
            d_optimizer.step()

            # ========== Train Generator ==========
            generator.zero_grad(set_to_none=True)

            fake_pred_for_g = discriminator(fake_images)
            g_loss = g_loss_fn(fake_pred_for_g.view(-1))
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        epoch_time = time.time() - epoch_start_time
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)

        print(f"Epoch [{epoch+1}/{epochs}] | Time: {epoch_time:.2f}s | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(ckpt_path, generator, discriminator, g_optimizer, d_optimizer, epoch)

    # Final save
    final_ckpt_path = os.path.join(output_dir, "checkpoint_final.pth")
    save_checkpoint(final_ckpt_path, generator, discriminator, g_optimizer, d_optimizer, epochs)
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to processed texture images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a .pth checkpoint to resume.')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs.')
    args = parser.parse_args()

    train_gan(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        image_size=args.image_size,
        checkpoint_path=args.checkpoint_path,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()
