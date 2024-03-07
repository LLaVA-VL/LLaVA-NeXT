#!/bin/bash

cd /mnt/bn/vl-research/workspace/boli01/zzzprojects/LLaVA

# Install yolk3k if not installed
if ! pip show yolk3k > /dev/null 2>&1; then
    pip install yolk3k
fi

# Get the installed version of transformers
installed_version=$(pip show transformers | grep Version | cut -d ' ' -f 2)

# Get the latest version of transformers from PyPI
latest_version=$(yolk -V transformers | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "$latest_version" ]; then
    pip install -U transformers
fi

# Get the installed version of deepspeed
installed_version=$(pip show deepspeed | grep Version | cut -d ' ' -f 2)

# Get the latest version of deepspeed from PyPI
latest_version=$(yolk -V deepspeed | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "$latest_version" ]; then
    pip install deepspeed==0.12.2
fi

# Install yolk3k if not installed
if ! pip show flash-attn > /dev/null 2>&1; then
    pip install flash-attn --no-build-isolation
fi


################## MISTRAL ##################
PROMPT_VERSION=mistral_instruct
MODEL_VERSION="Mistral-7B-Instruct-v0.2"
################## VICUNA ##################


################## project ##################
PROJECT_NAME="ds_llava-Mistral-7B-Instruct-v0.2-mlp2x_gelu-pretrain_blip558k_plain"

################## data ##################
DATA_NAME="llava_instruct_150k"

# wandb configure
export WANDB_API_KEY="03fc62d68025c9498cf6493432551badd7d4f953"
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME--$MODEL_VERSION--$DATA_NAME

export WANDB_PROJECT=LLaVA_Mixtral

export WANDB_MODE=online

wandb online


deepspeed --master_port 26000 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/$DATA_NAME.json \
    --image_folder /mnt/bn/vl-research/workspace/boli01/data/playground/data/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/$PROJECT_NAME/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava--$PROJECT_NAME--$MODEL_VERSION--$DATA_NAME--finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
