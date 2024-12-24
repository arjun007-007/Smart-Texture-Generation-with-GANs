# src/training/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class TextureDataset(Dataset):
    """
    A custom dataset for loading texture images for GAN training.
    Expects a directory of images (e.g., .png, .jpg).
    """
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 extensions=('.png', '.jpg', '.jpeg', '.tga', '.exr')):
        """
        Args:
            root_dir (str): Path to the directory containing texture images.
            transform (callable, optional): Transform to be applied on each image.
            extensions (tuple, optional): Allowed file extensions.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions

        # Gather all image file paths
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir} with extensions {extensions}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a single image (as a tensor).
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image
