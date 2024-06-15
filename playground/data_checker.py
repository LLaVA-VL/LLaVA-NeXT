import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml

def check_image_existence(args):
    image_root, data = args
    if "image" in data:
        full_image_path = os.path.join(image_root, data["image"])
        if not os.path.exists(full_image_path):
            print(f"WARNING!!! {full_image_path} not exists !!!")

    if data["conversations"][0]["value"].count("<image>") > 1:
        print(f"WARNING!!! {data['conversations'][0]['value']} has more than one <image> !!!")


def process_json_file(json_path, image_root):
    with open(json_path, "r") as f:
        data = json.load(f)

    current_items_count = len(data)
    # Prepare a list of tuples for the multiprocessing pool
    args = [(image_root, d) for d in data]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(check_image_existence, args), total=len(data)))

    del data
    return current_items_count


mode = "yaml"
image_root = "/mnt/bn/vl-research-cn-boli01-hl/data/llava_data"
total_items_count = 0

if mode == "json":
    json_path_list = [
        "/mnt/bn/vl-research/data/llava_instruct/dpo_data/rlaif_v_dataset.json",
    ]
    for json_path in json_path_list:
        current_items_count = process_json_file(json_path, image_root)
        total_items_count += current_items_count
        print(f"Processed {current_items_count} items from {json_path}")

    print(f"Total items processed: {total_items_count}")


elif mode == "yaml":
    yaml_path = "/mnt/bn/vl-research-cn-boli01-hl/workspace/boli01/projects/LLaVA_Next/scripts/cn_boli01_hl/config/final_stage_cauldron.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        for d in data["datasets"]:
            dd_json_path = d["json_path"]
            current_items_count = process_json_file(dd_json_path, image_root)

            sampling_strategy = d["sampling_strategy"]
            try:
                if sampling_strategy != "all":
                    percentage = sampling_strategy.split(":")[-1].replace("%", "") / 100.0
                    print(f"Sampling strategy: {sampling_strategy}")
                    print(f"Sampling percentage: {percentage}")
                    print(f"Sampling count: {int(current_items_count * percentage)}")
                else:
                    percentage = 1.0
            except Exception as e:
                print(f"Error: {e}")
                print(f"Sampling strategy: {sampling_strategy}")
                percentage = 1.0

            sampling_count = int(current_items_count * percentage)
            total_items_count += sampling_count
            print(f"Processed {sampling_count} items from {dd_json_path}")

    print(f"Total items processed: {total_items_count}")
