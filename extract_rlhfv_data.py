import os
import io
import json
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import ast

# 假设 data 是你的数据列表

dataset = load_dataset('parquet', data_files='/remote-home1/share/data/RLHF-V-Dataset/*parquet') 

print(len(dataset['train']))
# 创建保存图片的文件夹
image_dir = '/remote-home1/share/data/RLHF-V-Dataset/images'
os.makedirs(image_dir, exist_ok=True)

data = []
# 遍历数据，保存图片并更新路径

for item in tqdm(dataset['train']):
    # 保存图片到指定路径
    image_dir = f'/remote-home1/share/data/RLHF-V-Dataset/images'
    os.makedirs(image_dir, exist_ok=True)
    image_name = f"{item['idx']}.jpg"
    image_path = os.path.join(image_dir, image_name)
    
    if isinstance(item['image'], dict):
        image_bytes = item['image']['bytes']
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        # 检查图像模式，如果是 'P' 模式，则转换为 'RGB'
        if image.mode == 'P':
            image = image.convert('RGB')
        image.save(image_path, format='JPEG')
    else:
        item['image'].save(image_path, format='JPEG')

    
    # 更新字典中 'image' 字段为图片的路径
    item['image'] = image_name
    
    item['text'] = ast.literal_eval(item['text'])
    try:
        item['prompt'] = item['text']['question']
        item['chosen'] = item['text']['chosen']
        item['rejected'] = item['text']['rejected']
    except:
        import ipdb; ipdb.set_trace()
    
    # 去除字典中不需要的字段
    item = {k: v for k, v in item.items() if k in ['image', 'prompt', 'chosen', 'rejected']}
    
    data.append(item)

# 将处理后的数据保存为 JSON 文件
output_path = '/remote-home1/share/data/RLHF-V-Dataset/rlhfv.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"数据已保存至 {output_path}")