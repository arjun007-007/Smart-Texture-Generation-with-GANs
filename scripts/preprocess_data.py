# scripts/preprocess_data.py

import os
import argparse
from src.utils import file_io, image_ops

def preprocess_polyhaven_data(input_dir, output_dir, 
                              target_size=512, 
                              valid_extensions=('.png', '.jpg', '.jpeg', '.tga')):
    """
    Preprocess raw Poly Haven textures:
      - Ensure consistent resolution (e.g., 512x512).
      - Convert to PNG (optionally).
      - You can add more advanced steps (making them tileable, etc.).
    """
    file_io.ensure_dir_exists(output_dir)

    image_files = file_io.list_files_in_dir(input_dir, valid_extensions)

    for img_path in image_files:
        # Load image
        image = file_io.load_image(img_path)
        if image is None:
            print(f"Warning: Could not load {img_path}. Skipping.")
            continue
        
        # Resize / Crop
        image = image_ops.resize_image(image, (target_size, target_size))

        # Optional: Additional operations (e.g., color correction, tileability fixes)
        # image = image_ops.make_tileable(image) # If you implement something like that

        # Save processed image
        file_name = os.path.basename(img_path)
        name, _ = os.path.splitext(file_name)
        out_path = os.path.join(output_dir, f"{name}.png")
        file_io.save_image(image, out_path, format='PNG')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to raw Poly Haven textures.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save processed textures.')
    parser.add_argument('--size', type=int, default=512,
                        help='Target width/height for resizing.')
    args = parser.parse_args()

    preprocess_polyhaven_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.size
    )

if __name__ == "__main__":
    main()
