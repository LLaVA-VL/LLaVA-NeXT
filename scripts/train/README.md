# About Training Scripts

We first release the basic training scripts for LLaVA-NeXT. It's based on previous LLaVA's training scripts and researchers familiar with LLaVA will find it easy to use.

We will gradually release the more detailed training scripts for our LLaVA OneVision models including the mid stage, single-image final stage and one-vision final stage.
> They are basically the same as the basic training scripts, but with some modifications, such as the data yaml.

- `finetune_clip.sh`: This could be seen as the first image version LLaVA-NeXT (2024-01) training script, with `anyres` strategy and maximum 2x2 image grids.
- `finetune_siglip.sh`: Same but with `siglip` encoder, each grid becomes 729 tokens.
- `finetune_onevision.sh`: This is our latest training script, with `anyres_max_9` strategy and image grids weaving from 1x1 to 6x6, at most to 2304x2304 resolution. Inside the script, we also incorporate the multi-image and video data into training loop. the detail token strategy could be found in our paper.

# About the LLaVA-OneVision Data

We sincerely apologize for any inconvenience caused by the fact that our data has been used in different projects and has been collected and managed by different people. LLaVA-OneVision is our first attempt to integrate these datasets. For the data that has already been uploaded, we will refer you to the corresponding locations. We kindly ask everyone to gather the "fragments" and piece them together into a "diamond" in your own environment. 

Here we explain the some technical details on our data. 

- pretrain data - BLIP558K (same as previous llava 1.5 series)
- mid stage data mixture
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
  The first three datasets can be collected via the [LLaVA-Recap](https://huggingface.co/collections/lmms-lab/llava-next-6623288e2d61edba3ddbf5ff) series data. We did a slightly modification to make the data more compatible to other data with an added prompt at each question. For example:
- single-image stage data mixture
- onevision stage data mixture
