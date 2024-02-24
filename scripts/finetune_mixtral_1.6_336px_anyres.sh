#!/bin/bash
dataset_name=$1

cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next

# Install yolk3k if not installed
if ! pip show yolk3k > /dev/null 2>&1; then
    pip install yolk3k
fi

pip install pydantic

# Get the installed version of transformers
installed_version=$(pip show transformers | grep Version | cut -d ' ' -f 2)

# Get the latest version of transformers from PyPI
latest_version=$(yolk -V transformers | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.36.2" ]; then
    pip install transformers==4.36.2
fi

# Get the installed version of deepspeed
installed_version=$(pip show deepspeed | grep Version | cut -d ' ' -f 2)


# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.2" ]; then
    pip install deepspeed==0.12.2
fi

# Install flash-atten if not installed
if ! pip show flash-attn > /dev/null 2>&1; then
    pip install flash-attn --no-build-isolation
fi

################## MISTRAL ##################
PROMPT_VERSION=mistral_instruct
MODEL_VERSION="Mistral-7B-Instruct-v0.2"
################## MISTRAL ##################


################## project ##################
PROJECT_NAME="ds_llava-Mistral-7B-Instruct-v0.2-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain"

################## data ##################
DATA_NAME=$dataset_name


# wandb configure
export WANDB_API_KEY=e464cc107357c7b38e87f239bc3eb2ce5fb73c7c
export WANDB_PROJECT=llava

export WANDB_NAME=$PROJECT_NAME--$DATA_NAME--336px--anyres--sft

export WANDB_MODE=online

wandb online

deepspeed --master_port 26000 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/bn/vl-research/workspace/project/2023/LLaVA/checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/$DATA_NAME.json \
    --image_folder /mnt/bn/vl-research/workspace/boli01/data/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /mnt/bn/vl-research/workspace/project/2023/LLaVA/checkpoints/ds_llava-Mistral-7B-Instruct-v0.2-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./checkpoints/$PROJECT_NAME--$DATA_NAME--336px--anyres--sft \
    --num_train_epochs 9 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1500 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

