# src/utils/visualization.py

import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

def show_images(images, nrow=4, title=None, figsize=(10, 10), save_path=None):
    """
    Display or save a grid of images (tensor or PIL images).
    Args:
        images (Tensor or list of PIL Images): The images to display.
        nrow (int): Number of images in each row of the grid.
        title (str): Optional title for the figure.
        figsize (tuple): Size of the figure (in inches).
        save_path (str): If provided, saves the image grid instead of showing.
    """
    # If 'images' is a list of PIL images, convert them to a single batch tensor
    if isinstance(images, list):
        images = [vutils.to_tensor(img) for img in images]
        images = torch.stack(images, dim=0)

    # Make a grid from a 4D tensor (B, C, H, W)
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.axis('off')
    
    # Convert the tensor image to a numpy array and display
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
