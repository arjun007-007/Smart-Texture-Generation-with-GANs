# scripts/export_to_unreal.py

"""
Export generated textures to an Unreal Engine project.
By default, this script copies .png files from a specified source
to an Unreal project’s Content/Textures folder.
"""

import os
import shutil
import argparse

def export_to_unreal(source_dir, unreal_project_dir, texture_folder="Textures", valid_extensions=(".png", ".jpg", ".jpeg")):
    """
    Copies texture files from source_dir to UnrealEngineProject/Content/<texture_folder>/.
    Args:
        source_dir (str): Directory containing generated textures (e.g. .png).
        unreal_project_dir (str): Path to your Unreal project root.
        texture_folder (str): Folder under 'Content' where textures will be placed.
        valid_extensions (tuple): Allowed file extensions.
    """
    content_path = os.path.join(unreal_project_dir, "Content", texture_folder)
    os.makedirs(content_path, exist_ok=True)

    files_exported = 0
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith(valid_extensions):
            src_path = os.path.join(source_dir, file_name)
            dst_path = os.path.join(content_path, file_name)
            shutil.copy2(src_path, dst_path)
            files_exported += 1

    print(f"Export completed. {files_exported} texture(s) copied to {content_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True, help='Directory with generated textures.')
    parser.add_argument('--unreal_project_dir', type=str, required=True, help='Root directory of the Unreal project.')
    parser.add_argument('--texture_folder', type=str, default='Textures', help='Folder name under Content/.')
    args = parser.parse_args()

    export_to_unreal(
        source_dir=args.source_dir,
        unreal_project_dir=args.unreal_project_dir,
        texture_folder=args.texture_folder
    )

if __name__ == "__main__":
    main()
