import json
import os
import argparse
from tqdm import tqdm
import yaml


def check_missing_images(json_path, images_folder):
    data = json.load(open(json_path, "r"))
    missing_data = []

    for i, d in enumerate(tqdm(data)):
        image = d["image"] if "image" in d else ""
        if image != "":
            path = os.path.join(images_folder, image)
            if not os.path.exists(path):
                print(f"Missing image: {path}")
                missing_data.append(d)

    return missing_data


def read_yaml_to_llava_data(yaml_path, images_folder):
    print(f"Reading YAML file: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    llava_json_paths = data["datasets"]
    for item in llava_json_paths:
        json_path = item["json_path"]
        missing_data = check_missing_images(json_path, images_folder)
        if len(missing_data) > 0:
            print(f"Missing images in {json_path}:")
            for d in missing_data:
                print(d)


def direct_check_llava_data(json_path, images_folder):
    missing_data = check_missing_images(json_path, images_folder)
    if len(missing_data) > 0:
        print(f"Missing images in {json_path}:")
        for d in missing_data:
            print(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for missing images in dataset.")
    parser.add_argument("--yaml_path", type=str, default="", help="Path to the YAML file containing the dataset.")
    parser.add_argument("--json_path", type=str, default="", help="Path to the JSON file containing the dataset.")
    parser.add_argument("--images_folder", type=str, default="/mnt/bn/vl-research/data/llava_data", help="Path to the folder containing the images.")

    args = parser.parse_args()

    if args.json_path != "":
        direct_check_llava_data(args.json_path, args.images_folder)
    elif args.yaml_path != "":
        read_yaml_to_llava_data(args.yaml_path, args.images_folder)
