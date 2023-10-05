import argparse
import json
import os
import shutil

from pycocotools.coco import COCO


def split_coco_dataset_single_class(
    ann_file: str, image_dir: str, output_dir: str
) -> None:
    """
    Splits a COCO dataset into different class folders based on annotations for images with a single class.

    Args:
        ann_file (str): Path to the COCO annotation JSON file.
        image_dir (str): Path to the directory containing images.
        output_dir (str): Output directory for class folders.

    Returns:
        None
    """
    # Load the COCO annotation file
    coco = COCO(ann_file)

    # Create a dictionary to keep track of images with a single class
    single_class_images = {}

    # Iterate through annotations
    for ann in coco.dataset["annotations"]:
        image_id = ann["image_id"]
        class_id = ann["category_id"]

        # Check if the image has already been associated with a class
        if image_id not in single_class_images:
            single_class_images[image_id] = class_id
        else:
            # If the image already has a class, mark it as -1 for multiple classes
            single_class_images[image_id] = -1

    # Create class folders
    os.makedirs(output_dir, exist_ok=True)

    # Copy images to class folders
    for image_id, class_id in single_class_images.items():
        # Skip images with multiple classes (class_id == -1)
        if class_id == -1:
            continue

        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info["file_name"])
        class_name = coco.loadCats(class_id)[0]["name"]
        destination_folder = os.path.join(output_dir, class_name)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(
            image_path, os.path.join(destination_folder, image_info["file_name"])
        )


def main() -> None:
    """
    Main function to run the COCO dataset splitting script for single-class images.
    """
    parser = argparse.ArgumentParser(
        description="Split a COCO dataset into class folders based on annotations for single-class images."
    )
    parser.add_argument(
        "--annotation_file", required=True, help="Path to the COCO annotation JSON file"
    )
    parser.add_argument(
        "--image_dir", required=True, help="Path to the directory containing images"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for class folders"
    )

    args = parser.parse_args()

    split_coco_dataset_single_class(
        args.annotation_file, args.image_dir, args.output_dir
    )


if __name__ == "__main__":
    main()
