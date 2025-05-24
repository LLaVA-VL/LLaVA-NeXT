# About Training Scripts

🔧 [2025-05] Release Update: Clarification on Missing Files

We have received feedback regarding missing files in the current lmms-lab/LLaVA-OneVision-Data release. Specifically, the following datasets referenced in our config or documentation are currently not included in the repository:

- llava_wild_4v_39k.json
- llava_wild_4v_12k.json
- MathV360K_TQA_10181.json
- MathV360K_VQA-AS_5907.json
- MathV360K_VQA-RAD_2130.json
- Evol-Instruct-GPT4-Turbo-143000.json

We first release the basic training scripts for LLaVA-NeXT. It's based on previous LLaVA's training scripts and researchers familiar with LLaVA will find it easy to use.

We will gradually release the more detailed training scripts for our LLaVA OneVision models including the mid stage, single-image final stage and one-vision final stage.
> They are basically the same as the basic training scripts, but with some modifications, such as the data yaml.

- `finetune_clip.sh`: This could be seen as the first image version LLaVA-NeXT (2024-01) training script, with `anyres` strategy and maximum 2x2 image grids.
- `finetune_siglip.sh`: Same but with `siglip` encoder, each grid becomes 729 tokens.
- `finetune_onevision.sh`: This is our latest training script, with `anyres_max_9` strategy and image grids weaving from 1x1 to 6x6, at most to 2304x2304 resolution. Inside the script, we also incorporate the multi-image and video data into training loop. the detail token strategy could be found in our paper.

# About the LLaVA-OneVision Data

We need to address the fact that our data has been collected and used in different projects/people. LLaVA-OneVision is our first attempt to integrate these datasets. For the data that has already been uploaded, we will refer you to the corresponding locations. We kindly ask everyone to gather the "fragments" and piece them together into a "diamond" in your own environment. 

Here we explain the some technical details on our data. 

- **pretrain data** - BLIP558K (same as previous llava 1.5 series)
- **mid stage data mixture**
  ```yaml
    datasets:
      - json_path: /mnt/bn/vl-research/data/llava_instruct/blip558k_stage1.5_finetune_w_prompt.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/coco118k_stage1.5_finetune_w_prompt.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/cc3m_recap_data_prompt_v2.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/ureader_tr_sft.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/instruct_azure_dc_zh_92K.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/allava/Evol-Instruct-GPT4-Turbo-143K.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/synthdog_zh/synthdog_zh_100k.json
        sampling_strategy: all
      - json_path: /mnt/bn/vl-research/data/llava_instruct/synthdog_en/synthdog_en_100k.json
        sampling_strategy: all  
  ```
  The first three datasets can be collected via the [LLaVA-Recap](https://huggingface.co/collections/lmms-lab/llava-next-6623288e2d61edba3ddbf5ff) series data. We did a slightly modification to make the data more compatible to other data with an added prompt behind `<image>` at each question. You may download our dataset and process it in local environment. Each data we added the **same prompt** "\nPlease generate detailed descriptions of the given image.". Here's an example:
  ```json
    {
    "id": "000000000009",
    "image": "coco/train2017/000000000009.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease generate detailed descriptions of the given image."
      },
      {
        "from": "gpt",
        "value": "The image shows a meal served in a blue tray with compartments. In the top left compartment, there is a slice of bread with a spread that appears to be butter, accompanied by a few almonds and a slice of what looks like a baked potato or sweet potato. The top right compartment contains a variety of fruits, including what seems to be pineapple, orange slices, and possibly a piece of melon.\n\nIn the bottom left compartment, there is a piece of bread with a spread that could be butter or margarine, and a small portion of what might be a meatball or a similar type of meat covered in a sauce. The bottom right compartment contains a serving of broccoli, which appears to be steamed or lightly cooked.\n\nThe meal is presented in a way that suggests it is a balanced meal, with a variety of food groups represented carbohydrates (bread), protein (meatball), healthy fats (almonds and butter), and fruits and vegetables (broccoli and the fruit assortment). The image is a close-up photograph with a focus on the food, and the colors are vibrant, indicating freshness."
      }
    ]
  }
  ```

  The rest files are being uploaded to [LLaVA-OneVision Mid-Stage Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data), except from `instruct_azure_dc_zh_92K` since it's internal data due and have policy issue (sad but I could preview one example). The `instruct_azure_dc_zh_92K` images are from COCO and QAs are generated by GPT-4V service provided by AzureAI. We uses this data in early stage experiments wishing to add Chinese knowledge to our model, and just kept using it (yes, just legacy, no other specific reason).
  ```json
    {
    "id": "000000051920",
    "image": "azure_dc_zh_92K/000000051920.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n这幅画中的关键信息有哪些？"
      },
      {
        "from": "gpt",
        "value": "这张图片展示了一位穿着深色（可能是黑色或深蓝色）毛衣的人坐在一张桌子前，正在用右手拿着一片披萨。披萨看起来有多种配料，包括绿色的菠菜和橙红色的番茄，放在一个银色的圆形披萨盘上。披萨盘上还剩下大约六片披萨。在背景中，可以看到一个穿着浅色衣服的人站在柜台后面，柜台上方有一个黑板菜单，上面用白色粉笔写着菜单项，但是字迹不清晰，无法阅读。整个场景看起来像是在一个披萨店内部，环境显得比较休闲。"
      }
    ],
    "data_source": "azure_dc_zh_92K"
  }
  ```

- **single-image stage data mixture**

We have placed our single-image stage data in [single-image-yaml](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/single_image.yaml) for users to review. You can download each subset from [onevision-data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data). 

Inside the data yaml, the first indicates the previous llava-1.6/next 790K data, you can download them in [llava-next-data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data).

Inside the yaml, the naming would be different with our paper figure due to writing consideration. For users who need to explore our dataset, you can check the [upload script](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/0070d0ae4931c9b19d9cc57c38e16a87c270a61c/playground/upload_data.py#L175) to find the mapping from our local dataset to HF's version.

- **onevision stage data mixture**

Our onevision stage data is available in [onevision-yaml](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/onevision.yaml). The single-image portion can be downloaded from the above Huggingface link for onevision data. Here's a breakdown of each part:

  - Around 800K higher-quality data re-sampled from the previous stage (yes, it's data replay!).
  - Multi-image data is released in [M4-Instruct Data](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data). We combine the different subsets into two jsons (as they are mainly from DEMON and Mantis) in our training yaml, the jsons are:
    - /mnt/bn/vl-research/data/llava_instruct/real_vision_flan/llava_ofa_DEMON-FULL_filtered_311085.json
    - /mnt/bn/vl-research/data/llava_instruct/real_vision_flan/llava_ofa_mantis-instruct_reformatted.json
 
  - Video Data: We have released the video part along with [llava-video-data](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K). Users can download the data, and we utilize the subset used in LLaVA-OneVision:
    - We have included captions and open-ended questions in the `0_30_s_academic_v0_1` split, along with 240,000 open-ended QA items and 15,000 caption entries, as part of the video data in LLaVA-Hound for LLaVA-OneVision.
    - 0_30_s_academic_v0_1 captions
    - 0_30_s_academic_v0_1 open-ended QA
    - LLaVA-Hound: Same as above.
