from datasets import Dataset, Features, Value, ClassLabel, Sequence, Image
import json
import PIL.Image as pil_image
from io import BytesIO
from tqdm import tqdm

json_path = "/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/chrome_writting_train_8835.json"
short_name = "chrome_writting"
def gen():
    with open(json_path, "r") as f:
        data = json.load(f)

    preview_index = 10
    for item in tqdm(data):
        if preview_index > 0:
            preview_index -= 1
            print(item)
            continue
        
        image_path = f"/mnt/bn/vl-research/data/llava_data/{item['image']}"
        try:
            with open(image_path, "rb") as img_file:
                image = pil_image.open(BytesIO(img_file.read()))
        except:
            print(f"Failed to load image {item['image']}")
            continue

        yield {"id": item["id"], "image": image, "conversations": item["conversations"], "data_source": short_name}


hf_dataset = Dataset.from_generator(generator=gen)
hf_dataset.push_to_hub("lmms-lab/LLaVA-OneVision-Data", config_name=short_name, split="train")
    
# /mnt/bn/vl-research/data/llava_instruct/real_vision_flan/llavar_gpt4_20k.json, 19800
# /mnt/bn/vl-research/data/llava_instruct/real_vision_flan/sroie_data_33626.json, 33626
# /mnt/bn/vl-research/data/llava_instruct/real_vision_flan/chrome_writting_train_8835.json, 8835