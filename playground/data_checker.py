import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml


class DataProcessor:
    def __init__(self, file_path, image_root):
        self.file_path = file_path
        self.image_root = image_root
        self.data = None
        self.load_data()

    def load_data(self):
        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        elif self.file_path.endswith(".yaml"):
            with open(self.file_path, "r") as f:
                self.data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format")

    def load_json_data(self, json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def check_image_existence(self, data):
        if "image" in data:
            full_image_path = os.path.join(self.image_root, data["image"])
            if not os.path.exists(full_image_path):
                print(f"WARNING!!! {full_image_path} not exists !!!")

        if data["conversations"][0]["value"].count("<image>") > 1:
            print(f"WARNING!!! {data['conversations'][0]['value']} has more than one <image> !!!")

    def process_images(self):
        args = [(self.image_root, d) for d in self.data]
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(self.check_image_existence, args), total=len(self.data)))

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
            return total_items_count


def main(file_path, image_root, operation):
    processor = DataProcessor(file_path, image_root)
    if operation == "check_images":
        processor.process_images()
    elif operation == "count_items":
        total_items = processor.count_items()
        print(f"Total items: {total_items}")
    else:
        raise ValueError("Unsupported operation")


if __name__ == "__main__":
    file_path = "/path/to/your/file.yaml"  # or .json
    image_root = "/mnt/bn/vl-research-cn-boli01-hl/data/llava_data"
    operation = "check_images"  # or 'count_items'
    main(file_path, image_root, operation)
