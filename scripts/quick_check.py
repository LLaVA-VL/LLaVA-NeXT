import json
import os
import argparse
from tqdm import tqdm


def check_missing_images(json_path, images_folder):
    data = json.load(open(json_path, "r"))
    missing_data = []

    for i, d in enumerate(tqdm(data)):
        image = d["image"] if "image" in d else ""
        if image != "":
            path = os.path.join(images_folder, image)
            if not os.path.exists(path):
                print(d)
                missing_data.append(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for missing images in dataset.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing the dataset.")
    parser.add_argument("--images_folder", type=str, required=True, help="Path to the folder containing the images.")

    args = parser.parse_args()

    check_missing_images(args.json_path, args.images_folder)
