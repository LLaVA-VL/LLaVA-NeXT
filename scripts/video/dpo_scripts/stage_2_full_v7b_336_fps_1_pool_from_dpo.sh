#!/bin/bash
cd /mnt/bn/vl-research/workspace/yhzhang/llava_next/

# Get the installed version of transformers
installed_version=$(pip show transformers | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.39.2" ]; then
    pip install transformers==4.39.2
fi

# Get the installed version of deepspeed
installed_version=$(pip show deepspeed | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.2" ]; then
    pip3 install deepspeed==0.12.2
fi

# Install ninja if not installed
if ! pip3 show ninja > /dev/null 2>&1; then
    pip3 install ninja
fi

# Install flash-atten if not installed
if ! pip3 show flash-attn > /dev/null 2>&1; then
    pip3 install flash-attn --no-build-isolation
fi

# Install decord if not installed
if ! pip3 show decord > /dev/null 2>&1; then
    pip3 install decord
fi

# Install protobuf if not installed
if ! pip3 show protobuf > /dev/null 2>&1; then
    pip3 install protobuf 
fi

# Install torchvision if not installed
if ! pip3 show torchvision > /dev/null 2>&1; then
    pip3 install torchvision==0.16.0
fi

# Install timm if not installed
if ! pip3 show timm > /dev/null 2>&1; then
    pip3 install timm
fi

# Install datasets if not installed
if ! pip3 show datasets > /dev/null 2>&1; then
    pip3 install datasets
fi

# Install tyro if not installed
if ! pip3 show tyro > /dev/null 2>&1; then
    pip3 install tyro
fi

# Get the installed version of transformers
installed_version=$(pip3 show accelerate | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.27.2" ]; then
    pip3 install accelerate==0.27.2
fi


# Install sentencepiece if not installed
if ! pip3 show sentencepiece > /dev/null 2>&1; then
    pip3 install sentencepiece==0.1.99
fi

# Install opencv if not installed
if ! pip3 show opencv-python-headless > /dev/null 2>&1; then
    pip3 install opencv-python-headless

fi
pip install -U wandb

CONV_MODE=$1
FRAMES_UPBOUND=$2
STRIDE=$3
MODEL_MAX_LENGTH=$4
SFT_MODEL=$5

################## project ##################
PROJECT_NAME=llava-vicuna_7B-mlp2x_gelu-llava_558k-336px-frames_upbound_${FRAMES_UPBOUND}-stride_${STRIDE}-model_max_length_${MODEL_MAX_LENGTH}-dpo-test


# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb online

gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

torchrun --nproc_per_node=$n_gpu --master_port=30000 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $SFT_MODEL \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version $CONV_MODE \
    --data_path  /mnt/bn/vl-research/data/llava_video/shareVideoGPTV/dpo/sft_dpo_17k.jsonl \
    --image_folder /mnt/bn/vl-research/data/llava_data/ \
    --video_folder /mnt/bn/vl-research/data/llava_video/shareVideoGPTV/frames/all_frames/ \
    --vision_tower  openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --frames_upbound ${FRAMES_UPBOUND:-0} \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./work_dirs/$PROJECT_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 70 \
    --save_total_limit 11 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_stride ${STRIDE} \
    --mm_spatial_pool_out_channels 1024 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --freeze_mm_mlp_adapter True


