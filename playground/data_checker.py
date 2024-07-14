import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml


class DataProcessor:
    def __init__(self, file_path, image_root, video_root):
        self.file_path = file_path
        self.image_root = image_root
        self.data = None
        self.video_root = video_root
        self.load_data()

    def load_data(self):
        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        elif self.file_path.endswith(".yaml"):
            with open(self.file_path, "r") as f:
                self.data = yaml.safe_load(f)
        elif self.file_path.endswith(".jsonl"):
            with open(self.file_path, "r") as f:
                self.data = [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError("Unsupported file format")

    def load_json_data(self, json_path):
        if json_path.endswith(".jsonl"):
            cur_data_dict = []
            with open(json_path, "r") as json_file:
                for line in json_file:
                    cur_data_dict.append(json.loads(line.strip()))
            return cur_data_dict
        elif json_path.endswith(".json"):
            with open(json_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported file format")

    def check_image_existence(self, data):
        if "image" in data:
            if type(data["image"]) == list:
                images = data["image"]
            else:
                images = [data["image"]]

            for image in images:
                full_image_path = os.path.join(self.image_root, image)
                if not os.path.exists(full_image_path):
                    print(f"WARNING!!! {full_image_path} not exists !!!")

        if "video" in data:
            full_video_path = os.path.join(self.video_root, data["video"])
            if not os.path.exists(full_video_path):
                print(f"WARNING!!! {full_video_path} not exists !!!")

        # if data["conversations"][0]["value"].count("<image>") > 1:
        #     print(f"WARNING!!! {data['conversations'][0]['value']} has more than one <image> !!!")

    def process_images(self):
        if isinstance(self.data, list):
            args = [d for d in self.data]
            with Pool(processes=cpu_count()) as pool:
                list(tqdm(pool.imap(self.check_image_existence, args), total=len(self.data)))
        elif isinstance(self.data, dict):
            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                args = [d for d in data]
                with Pool(processes=cpu_count()) as pool:
                    list(tqdm(pool.imap(self.check_image_existence, args), total=len(data), desc=f"Processing {dd_json_path}"))

    def count_items(self):
        if isinstance(self.data, list):  # Assuming JSON data loaded directly
            return len(self.data)
        elif isinstance(self.data, dict):  # Assuming YAML data loaded
            total_items_count = 0
            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                current_items_count = len(data)

                sampling_strategy = d["sampling_strategy"]
                try:
                    if sampling_strategy != "all":
                        percentage = float(sampling_strategy.split(":")[-1].replace("%", "")) / 100.0
                    else:
                        percentage = 1.0
                except Exception as e:
                    print(f"Error: {e}")
                    percentage = 1.0

                sampling_count = int(current_items_count * percentage)
                total_items_count += sampling_count
                print(f"{dd_json_path}: {sampling_count}")
            return total_items_count

    def stat_data(self):
        if isinstance(self.data, dict):
            cur_lens_list = []
            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                print(f"{dd_json_path}: {len(data)}")
                for item in data:
                    cur_len = sum([len(conv["value"]) for conv in item["conversations"]])

                    cur_lens_list.append(cur_len)
            print(f"Max length: {max(cur_lens_list)}, Min length: {min(cur_lens_list)}, Average length: {sum(cur_lens_list) / len(cur_lens_list)}")
                    

    def filter_data(self):
        if isinstance(self.data, dict):
            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                print(f"Processing {dd_json_path}")
                data = self.load_json_data(dd_json_path)

                filtered_data = []
                mismatch_data = []
                mismatch_flag = False
                for item in data:
                    if "image" in item:
                        num_image = len(item["image"]) if isinstance(item["image"], list) else 1
                    else:
                        num_image = 0

                    if "video" in item:
                        num_video = len(item["video"]) if isinstance(item["video"], list) else 1
                    else:
                        num_video = 0

                    num_visuals = num_image + num_video
                    conv_text = ""
                    for conv in item["conversations"]:
                        conv_text += conv["value"]

                    num_img_token_appearance = conv_text.count("<image>")
                    if len(conv_text) == 0:
                        print(f"Conversation text is empty for {item}")
                        
                    if num_img_token_appearance == num_visuals or num_img_token_appearance < num_visuals and len(conv_text) > 0:
                        filtered_data.append(item)
                    elif num_img_token_appearance > num_visuals:
                        item["num_img_token_appearance"] = num_img_token_appearance
                        item["num_visuals"] = num_visuals
                        mismatch_data.append(item)

                        if not mismatch_flag:
                            print(f"Data mismatch for {item}")

                        mismatch_flag = True

                if mismatch_flag:
                    print(f"Data mismatch for {dd_json_path}")

                if len(filtered_data) < len(data):
                    saving_dd_json_path = dd_json_path.replace(".jsonl", f"fltd_{len(filtered_data)}.json").replace(".json", f"fltd_{len(filtered_data)}.json")
                    with open(saving_dd_json_path, "w") as f:
                        json.dump(filtered_data, f, indent=2)
                    print(f"Filtered data count: {len(filtered_data)}")
                else:
                    pass


def main(file_path, image_root, operation, video_root):
    processor = DataProcessor(file_path, image_root, video_root)
    if operation == "check":
        processor.process_images()
    elif operation == "count":
        total_items = processor.count_items()
        print(f"Total items: {total_items}")
    elif operation == "filter":
        processor.filter_data()
    elif operation == "stat":
        processor.stat_data()
    else:
        raise ValueError("Unsupported operation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_continual.yaml")
    parser.add_argument("--image_root", type=str, default="/mnt/bn/vl-research/data/llava_data")
    parser.add_argument("--video_root", type=str, default="/mnt/bn/vl-research/data/llava_video")
    parser.add_argument("--operation", type=str, default="filter")
    args = parser.parse_args()
    main(args.file_path, args.image_root, args.operation, args.video_root)
