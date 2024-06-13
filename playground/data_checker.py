import json
import os
from random import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml

def check_image_existence(args):
    image_root, data = args
    if "image" in data:
        full_image_path = os.path.join(image_root, data["image"])
        if not os.path.exists(full_image_path):
            print(f"WARNING!!! {full_image_path} not exists !!!")

def process_json_file(json_path, image_root):
    with open(json_path, "r") as f:
        data = json.load(f)[:1000]

    # Prepare a list of tuples for the multiprocessing pool
    args = [(image_root, d) for d in data]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(check_image_existence, args), total=len(data)))


mode = "yaml"
image_root = "/mnt/bn/vl-research/data/llava_data"

if mode == "json":
    json_path_list = [
        "/mnt/bn/vl-research/data/llava_instruct/dpo_data/rlaif_v_dataset.json",
    ]

    for json_path in json_path_list:
        process_json_file(json_path, image_root)

elif mode == "yaml":
    yaml_path = "/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/mid_stage_original.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        for d in data["datasets"]:
            dd_json_path = d["json_path"]
            print(f"Processing {dd_json_path}")
            process_json_file(dd_json_path, image_root)
