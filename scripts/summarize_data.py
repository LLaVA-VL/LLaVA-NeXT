import json
import os
from tqdm import tqdm

with open("/mnt/bn/vl-research/workspace/boli01/zzzprojects/LLaVA/playground/data/llava_v1_5_mix665k.json") as f:
    llava_v1_5_mix665k = json.load(f)  # 665298

with open("/mnt/bn/vl-research/workspace/boli01/zzzprojects/LLaVA/playground/data/llava_instruct_150k.json") as f:
    llava_instruct_150k = json.load(f)  # 157712

# Create sets of "id" fields
mix665k_ids = set()
for item in llava_v1_5_mix665k:
    all_conv = ""
    for cur_conversation in item["conversations"]:
        all_conv += cur_conversation["value"]
    mix665k_ids.add(f'{item["id"]}_{all_conv}')

instruct_150k_ids = set()
for item in llava_instruct_150k:
    all_conv = ""
    for cur_conversation in item["conversations"]:
        all_conv += cur_conversation["value"]
    instruct_150k_ids.add(f'{item["id"]}_{all_conv}')

share_gpt_ids = set()
for item in llava_v1_5_mix665k:
    if "image" not in item:
        all_conv = ""
        for cur_conversation in item["conversations"]:
            all_conv += cur_conversation["value"]
        share_gpt_ids.add(f'{item["id"]}_{all_conv}')  # 40688

# Get "id" fields that are in mix665k but not in instruct_150k and share_gpt
new_ids = mix665k_ids - instruct_150k_ids - share_gpt_ids  # 466898

# Get "id" fields that are in mix665k but not in share_gpt
# new_ids = mix665k_ids - share_gpt_ids #624610

# import pdb; pdb.set_trace()

# Filter mix665k data based on new_ids
new_data = []
for item in llava_v1_5_mix665k:
    all_conv = ""
    for cur_conversation in item["conversations"]:
        all_conv += cur_conversation["value"]
    if f'{item["id"]}_{all_conv}' in new_ids:
        new_data.append(item)

import pdb

pdb.set_trace()

with open("/mnt/bn/vl-research/workspace/boli01/zzzprojects/LLaVA/playground/data/mixtral_instruct_135K_of_158K_V1.5.json") as f:
    new_mixtral_instruct = json.load(f)

# mixtral_instruct_50K_of_80K_V1.json@

# print(len(new_data))
# for _ in new_mixtral_instruct:
#     # import pdb; pdb.set_trace()
#     if "coco" not in _["image"]:
#         _["image"] = f"coco/train2017/{_['image']}"
#     new_data.append(_)

# print(len(instruct_150k_ids))
print(len(new_data))

# for _ in tqdm(new_data):
#     if "image" in _:
#         if "000000442654" in _["image"]:
#             all_conv = ""
#             for cur_conversation in _["conversations"]:
#                 all_conv += cur_conversation["value"]
#         # if not os.path.exists(f'/mnt/bn/vl-research/workspace/boli01/data/playground/data/{_["image"]}'):
#             import pdb; pdb.set_trace()

# Write new_data to a new JSON file
with open("/mnt/bn/vl-research/workspace/boli01/zzzprojects/LLaVA/playground/data/llava_v1_5_mix665k_minus_llava_instruct_150k_minus_sharegpt_plus_mixtral_instruct_135K_of_158K_V1.5.json", "w") as f:
    json.dump(new_data, f)
