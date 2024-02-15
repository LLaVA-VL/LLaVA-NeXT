#!/bin/bash
dataset_name=$1

cd /mnt/sfs-common/libo/kaichen/LLaVA_Next

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
DATA_NAME='detail_1'


# wandb configure
export WANDB_API_KEY=e9f0dc0578376e9ce4e1303ae0346da601810f90
export WANDB_PROJECT=llava

export WANDB_NAME=$PROJECT_NAME--$DATA_NAME--336px--anyres--sft

export WANDB_MODE=online

wandb online

DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" deepspeed --master_port 26000 --include localhost:4,5,6,7\
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path mistralai/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /mnt/sfs-common/libo/kaichen/datasets/LLava/$DATA_NAME.json \
    --image_folder /mnt/sfs-common/libo/kaichen/datasets/COCO2017/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./checkpoints/$PROJECT_NAME--$DATA_NAME--336px--anyres--sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 1 \
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
    --report_to wandb \
    --eval_num_processes 4 \
    --task_names gqa,coco_val2017 \
    --model_args pretrained=./checkpoints/$PROJECT_NAME--$DATA_NAME--336px--anyres--sft \
    --limit 8 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix debug \
    --output_path ./logs/