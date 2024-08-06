# About Training Scripts

We first release the basic training scripts for LLaVA-NeXT. It's based on previous LLaVA's training scripts and researchers familiar with LLaVA will find it easy to use.

We will gradually release the more detailed training scripts for our LLaVA OneVision models including the mid stage, single-image final stage and one-vision final stage.
> They are basically the same as the basic training scripts, but with some modifications, such as the data yaml.

- `finetune_clip.sh`: This could be seen as the first image version LLaVA-NeXT (2024-01) training script, with `anyres` strategy and maximum 2x2 image grids.
- `finetune_siglip.sh`: Same but with `siglip` encoder, each grid becomes 729 tokens.
- `finetune_onevision.sh`: This is our latest training script, with `anyres_max_9` strategy and image grids weaving from 1x1 to 6x6, at most to 2304x2304 resolution. Inside the script, we also incorporate the multi-image and video data into training loop. the detail token strategy could be found in our paper.