#!/bin/bash

# set up wandb
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=llava-vl
export WANDB_PROJECT=llava-next
export PYTHONWARNINGS="ignore"

cd /mnt/bn/vl-research/workspace/boli01/projects/lmms-eval

pip install -e .

# set up llava dev env
cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next

################## MISTRAL ##################
PROMPT_VERSION=mistral_instruct
MODEL_VERSION="Mistral-7B-Instruct-v0.2"
################## MISTRAL ##################

################## project ##################
PROJECT_NAME="ds_llava-Mistral-7B-Instruct-v0.2-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain"

################## data ##################
DATA_NAME='llava_caps20k_chartqa19k'

export WANDB_NAME=$PROJECT_NAME--$DATA_NAME--336px--anyres--sft
export WANDB_MODE=online

wandb online

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" deepspeed --master_port 26000 --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path mistralai/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct/$DATA_NAME.json \
    --image_folder /mnt/bn/vl-research/data/llava \
    --vision_tower openai/clip-vit-large-patch14-336 \
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
    --output_dir ./checkpoints/$PROJECT_NAME--llava1.6--336px--anyres--sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $WANDB_NAME
# starting here is the args for evaluation
    --eval_num_processes 4 \ 
    --task_names mme,docvqa_val \
    --model_args pretrained=./checkpoints/$PROJECT_NAME--$DATA_NAME--336px--anyres--sft \
    --limit 8 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix debug \
    --output_path ./logs/
