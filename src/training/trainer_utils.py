# src/training/trainer_utils.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

def get_adversarial_loss_fn():
    """
    Returns the standard BCE-based adversarial loss for the discriminator and generator.
    """
    bce_loss = nn.BCEWithLogitsLoss()

    def d_loss_fn(real_pred, fake_pred):
        """
        Discriminator loss.
        real_pred: (batch_size,) - logits for real images
        fake_pred: (batch_size,) - logits for fake images
        Returns scalar loss for D.
        """
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        real_loss = bce_loss(real_pred, real_labels)
        fake_loss = bce_loss(fake_pred, fake_labels)
        return (real_loss + fake_loss) * 0.5

    def g_loss_fn(fake_pred):
        """
        Generator loss.
        fake_pred: (batch_size,) - logits for generated (fake) images
        Returns scalar loss for G.
        """
        real_labels = torch.ones_like(fake_pred)
        return bce_loss(fake_pred, real_labels)

    return d_loss_fn, g_loss_fn

def build_optimizers(generator, discriminator, lr=0.0002, betas=(0.5, 0.999)):
    """
    Build separate Adam optimizers for G and D.
    """
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    return g_optimizer, d_optimizer

def save_checkpoint(save_path, generator, discriminator, g_optimizer, d_optimizer, epoch):
    """
    Save the current model, optimizers, and epoch to a checkpoint file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(load_path, generator, discriminator, g_optimizer=None, d_optimizer=None):
    """
    Load model weights and (optionally) optimizer states from a checkpoint file.
    """
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint {load_path} not found.")

    checkpoint = torch.load(load_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    if g_optimizer is not None and d_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {load_path} at epoch {checkpoint['epoch']}")
    return start_epoch

def move_to_device(generator, discriminator):
    """
    Moves generator and discriminator to GPU if available; else CPU.
    Returns the device used.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    return device
