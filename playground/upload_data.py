from datasets import Dataset, Features, Value, ClassLabel, Sequence, Image
import json
import PIL.Image as pil_image
from io import BytesIO
from tqdm import tqdm


def gen():
    json_path = "/mnt/bn/vl-research/data/llava_instruct/cc3m_recap_data_simple.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    preview_index = 10
    for item in tqdm(data):
        if preview_index > 0:
            preview_index -= 1
            print(item)
            continue
        
        image_path = f"/mnt/bn/vl-research/data/{item['image']}"
        try:
            with open(image_path, "rb") as img_file:
                image = pil_image.open(BytesIO(img_file.read()))
        except:
            print(f"Failed to load image {item['image']}")
            continue

        yield {"id": item["id"], "image": image, "conversations": item["conversations"], "data_source": "llava_recap_cc3m"}


hf_dataset = Dataset.from_generator(generator=gen)
hf_dataset.push_to_hub("lmms-lab/LLaVA-ReCap-CC3M", token="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")
