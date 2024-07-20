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

    def check_item_structure(self, item):
        if not all(key in item for key in ["conversations"]):
            print(f"WARNING!!! Item {item.get('id', 'unknown')} is missing required fields!")
            return False

        conversations = item["conversations"]
        if not isinstance(conversations, list) or len(conversations) < 2 or len(conversations) % 2 != 0:
            print(f"WARNING!!! Item {item['id']} has invalid conversations structure!")
            return False

        for i, conv in enumerate(conversations):
            if not all(key in conv for key in ["from", "value"]):
                print(f"WARNING!!! Item {item['id']} has invalid conversation format!")
                return False

            expected_from = "human" if i % 2 == 0 else "gpt"
            if conv["from"] != expected_from:
                print(f"WARNING!!! Item {item['id']} has incorrect conversation order!")
                return False

        return True

    def check_image_and_structure(self, item):
        if not self.check_item_structure(item):
            return

        # self.check_image_existence(item)

    def process_images(self):
        if isinstance(self.data, list):
            args = [d for d in self.data]
            with Pool(processes=cpu_count()) as pool:
                list(tqdm(pool.imap(self.check_image_and_structure, args), total=len(self.data)))
        elif isinstance(self.data, dict):
            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                args = [d for d in data]
                with Pool(processes=cpu_count()) as pool:
                    list(tqdm(pool.imap(self.check_image_and_structure, args), total=len(data), desc=f"Processing {dd_json_path}"))

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
            single_image_count = 0
            multiple_image_count = 0
            video_count = 0
            total_count = 0
            text_count = 0
            max_tokens_item = None
            max_tokens = 0

            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                sampling_strategy = d["sampling_strategy"]

                try:
                    if sampling_strategy != "all":
                        percentage = float(sampling_strategy.split(":")[-1].replace("%", "")) / 100.0
                    else:
                        percentage = 1.0
                except Exception as e:
                    print(f"Error parsing sampling strategy: {e}")
                    percentage = 1.0

                sampled_count = int(len(data) * percentage)
                print(f"{dd_json_path}: {sampled_count} (sampled from {len(data)})")

                for item in data[:sampled_count]:
                    conversations = item["conversations"]
                    cur_len = sum([len(conv["value"].split()) for conv in conversations])
                    cur_lens_list.append(cur_len)

                    if cur_len > max_tokens:
                        max_tokens = cur_len
                        max_tokens_item = item

                    total_count += 1
                    if "image" in item:
                        if isinstance(item["image"], list):
                            if len(item["image"]) > 1:
                                multiple_image_count += 1
                            else:
                                single_image_count += 1
                        else:
                            single_image_count += 1
                    elif "video" in item:
                        video_count += 1
                    else:
                        text_count += 1

            print(f"Max length: {max(cur_lens_list)}, Min length: {min(cur_lens_list)}, Average length: {sum(cur_lens_list) / len(cur_lens_list)}")
            print(f"Total items: {total_count}")
            print(f"Text items: {text_count} ({text_count/total_count*100:.2f}%)")
            print(f"Single image items: {single_image_count} ({single_image_count/total_count*100:.2f}%)")
            print(f"Multiple image items: {multiple_image_count} ({multiple_image_count/total_count*100:.2f}%)")
            print(f"Video items: {video_count} ({video_count/total_count*100:.2f}%)")

            print("\nItem with the largest number of tokens:")
            print(f"Token count: {max_tokens}")
            print("Item content:")
            print(json.dumps(max_tokens_item, indent=2))

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
                    try:
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
                    except Exception as e:
                        print(f"Error: {e}")
                        print()

                if mismatch_flag:
                    print(f"Data mismatch for {dd_json_path}")

                if len(filtered_data) < len(data):
                    saving_dd_json_path = dd_json_path.replace(".jsonl", f"fltd_{len(filtered_data)}.json").replace(".json", f"fltd_{len(filtered_data)}.json")
                    with open(saving_dd_json_path, "w") as f:
                        json.dump(filtered_data, f, indent=2)
                    print(f"Filtered data count: {len(filtered_data)}")
                else:
                    pass

    def stat_and_filter_data(self, threshold):
        if isinstance(self.data, dict):
            cur_lens_list = []
            single_image_count = 0
            multiple_image_count = 0
            video_count = 0
            total_count = 0
            text_count = 0

            for d in self.data["datasets"]:
                dd_json_path = d["json_path"]
                data = self.load_json_data(dd_json_path)
                sampling_strategy = d["sampling_strategy"]
                filtered_data = []

                try:
                    if sampling_strategy != "all":
                        percentage = float(sampling_strategy.split(":")[-1].replace("%", "")) / 100.0
                    else:
                        percentage = 1.0
                except Exception as e:
                    print(f"Error parsing sampling strategy: {e}")
                    percentage = 1.0

                sampled_count = int(len(data) * percentage)
                print(f"{dd_json_path}: {sampled_count} (sampled from {len(data)})")

                save_flag = False
                for item in data:
                    total_count += 1
                    conversations = item["conversations"]
                    filtered_conversations = []
                    current_token_count = 0

                    for i in range(0, len(conversations), 2):
                        if i + 1 < len(conversations):
                            human_conv = conversations[i]
                            gpt_conv = conversations[i + 1]
                            pair_tokens = len(human_conv["value"].split()) + len(gpt_conv["value"].split())

                            if current_token_count + pair_tokens <= threshold:
                                filtered_conversations.extend([human_conv, gpt_conv])
                                current_token_count += pair_tokens
                            else:
                                save_flag = True
                                break

                    if filtered_conversations:
                        item["conversations"] = filtered_conversations
                        cur_len = sum([len(conv["value"].split()) for conv in filtered_conversations])
                        cur_lens_list.append(cur_len)
                        filtered_data.append(item)

                        if "image" in item:
                            if isinstance(item["image"], list):
                                if len(item["image"]) > 1:
                                    multiple_image_count += 1
                                else:
                                    single_image_count += 1
                            else:
                                single_image_count += 1
                        elif "video" in item:
                            video_count += 1
                        else:
                            text_count += 1

                # Save filtered data for each dataset
                if filtered_data and save_flag:
                    if dd_json_path.endswith(".jsonl"):
                        output_file = dd_json_path.replace(".jsonl", f"_filtered_{threshold}tokens_{len(filtered_data)}.jsonl")
                        with open(output_file, "w") as f:
                            for item in filtered_data:
                                f.write(json.dumps(item) + "\n")
                    else:
                        output_file = dd_json_path.replace(".json", f"_filtered_{threshold}tokens_{len(filtered_data)}.json")
                        with open(output_file, "w") as f:
                            json.dump(filtered_data, f, indent=2)
                    print(f"Filtered data for {dd_json_path} saved to: {output_file}")

            print(f"Max length: {max(cur_lens_list)}, Min length: {min(cur_lens_list)}, Average length: {sum(cur_lens_list) / len(cur_lens_list)}")
            print(f"Total items: {total_count}")
            print(f"Text items: {text_count} ({text_count/total_count*100:.2f}%)")
            print(f"Single image items: {single_image_count} ({single_image_count/total_count*100:.2f}%)")
            print(f"Multiple image items: {multiple_image_count} ({multiple_image_count/total_count*100:.2f}%)")
            print(f"Video items: {video_count} ({video_count/total_count*100:.2f}%)")


def main(file_path, image_root, operation, video_root, threshold=None):
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
    elif operation == "stat_and_filter":
        if threshold is None:
            raise ValueError("Threshold must be provided for stat_and_filter operation")
        processor.stat_and_filter_data(threshold)
    else:
        raise ValueError("Unsupported operation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_continual.yaml")
    parser.add_argument("--image_root", type=str, default="/mnt/bn/vl-research/data/llava_data")
    parser.add_argument("--video_root", type=str, default="/mnt/bn/vl-research/data/llava_video")
    parser.add_argument("--operation", type=str, default="filter")
    parser.add_argument("--threshold", type=int, default=None, help="Threshold for stat_and_filter operation")
    args = parser.parse_args()
    main(args.file_path, args.image_root, args.operation, args.video_root, args.threshold)
