# src/utils/file_io.py

import os
from PIL import Image

def ensure_dir_exists(path):
    """
    Create the directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def list_files_in_dir(dir_path, valid_extensions=('.png', '.jpg', '.jpeg')):
    """
    Recursively list all files in dir_path with given extensions.
    """
    all_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_files.append(os.path.join(root, file))
    return all_files

def load_image(path):
    """
    Load an image from disk as a PIL Image object.
    Return None if there's an error.
    """
    try:
        img = Image.open(path)
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def save_image(image, path, format='PNG'):
    """
    Save a PIL Image object to disk in a given format.
    """
    try:
        image.save(path, format=format)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
