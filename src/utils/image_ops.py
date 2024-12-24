# src/utils/image_ops.py

from PIL import Image

def resize_image(image, size):
    """
    Resize PIL image to the given size (width, height).
    """
    return image.resize(size, Image.LANCZOS)

def center_crop(image, size):
    """
    Center-crop a PIL image to (width, height).
    """
    width, height = image.size
    target_w, target_h = size
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return image.crop((left, top, right, bottom))

def make_tileable(image):
    """
    Placeholder for advanced tile-making logic (seam removal, offset, etc.).
    Could integrate Gaussian Poisson blending or specialized seamless techniques.
    """
    # For demonstration, we will just return the original image.
    # In a real scenario, you'd implement your tiling logic here.
    return image

