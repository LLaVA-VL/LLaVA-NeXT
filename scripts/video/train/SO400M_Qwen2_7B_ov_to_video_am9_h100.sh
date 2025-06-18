#!/bin/bash

# ULTIMATE H100 Performance Training Script
# Set up the data folder
IMAGE_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"
VIDEO_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"

mkdir -p data
aws s3 cp s3://scalable-training-dataset/gemini_fine_tuning/32k/gemini_finetuning_subset_cheating_description.json data/gemini_finetuning_subset_cheating_description.json
DATA_YAML="data/gemini_finetuning_subset_cheating_description.json"

############### Prepare Envs #################
python3 -m pip install flash-attn --no-build-isolation
alias python=python3

############### Show Envs ####################
nvidia-smi

################ ULTIMATE H100 OPTIMIZATIONS ################
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# Auto-detect network interface
export NCCL_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}' | head -1)
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_READ=1
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_NET_SHARED_BUFFERS=0

# CUDA optimizations for H100
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True

# Compiler optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_ALLOW_TF32=1

################ Model Configuration ################
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9_h100"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Pass SSH options to deepspeed
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no"
mkdir -p ~/.ssh

# ULTIMATE H100 training with maximum optimizations
ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nnodes="${GPU_INSTANCES_NUMBER}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPU_COUNT}" \
    --master_addr="${MASTER_PRIVATE_IP}" \
    --master_port=1234 \
  llava/train/train_mem.py \
    --deepspeed scripts/zero3_h100_optimized.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "'(1x1),...,(6x6)'" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --torch_compile_mode "max-autotune" \
    --dataloader_drop_last True \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --frames_upbound 40 \
    --video_fps 5 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --mm_spatial_pool_stride 2 \
    --verbose_logging \
    --report_to tensorboard \
    --attn_implementation "flash_attention_2" \
    --max_grad_norm 1.0 \
    --ddp_timeout 18000 \
    --save_safetensors True

exit 0; 