#!/bin/bash
NAS_REGION="vl-research";
USER_PROJECT="boli01"

export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=llava-vl
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

export ACCELERATE_DEBUG_MODE="1"
export HF_HOME=/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface
export HF_TOKEN="HF_Token"
export HF_HUB_ENABLE_HF_TRANSFER="1"

############### Prepare Envs #################
cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next

git config --global --add safe.directory '*'

alias python=python3
############### Show Envs ####################
nvidia-smi
# 取 worker0 第一个 port
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}"| awk -F',' '{print $1}')"

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"
echo "master port in cmd: ${port_in_cmd}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO


PORT=26000
GPUS="0,1,2,3,4,5,6,7"

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb offline

installed_version=$(pip3 show transformers | grep Version | cut -d ' ' -f 2)

# # Check if the installed version is not the latest
if [ "$installed_version" != "4.39.2" ]; then
    pip3 install transformers==4.39.2
fi

# Get the installed version of deepspeed
installed_version=$(pip3 show deepspeed | grep Version | cut -d ' ' -f 2)

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
    pip3 install flash-attn --no-cache-dir --no-build-isolation
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
pip3 install -U wandb
pip3 install peft
pip3 install ftfy
pip3 install matplotlib
pip3 install scipy

CONV_MODE=qwen_1_5
FRAMES_UPBOUND=32
STRIDE=2
MODEL_MAX_LENGTH=32768
echo $((16 / ARNOLD_WORKER_NUM))


################## project ##################
PROJECT_NAME=llava-Qwen_72B-mlp2x_gelu-llava_558k-336px-frames_upbound_${FRAMES_UPBOUND}-stride_${STRIDE}-model_max_length_${MODEL_MAX_LENGTH}-dpo-dist${ARNOLD_WORKER_NUM}-trial3
SFT_MODEL=/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/project_checkpoints/llavanext-Qwen_Qwen1.5-72B-Chat-openai_clip-vit-large-patch14-336-pretrain_blip558k_plain-ft_la1_6mix_2e5-lr1e_5_ufvis_lr2e6_d32k_trial2
# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb online

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_dpo.py \
    --deepspeed ./scripts/zero3.json \
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
    --output_dir /mnt/bn/vl-research/checkpoints/$PROJECT_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $((16 / ARNOLD_WORKER_NUM)) \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_stride ${STRIDE} \
    --mm_spatial_pool_out_channels 1024 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --freeze_mm_mlp_adapter True 
    # --tune_mm_mlp_adapter True 
    # --freeze_mm_mlp_adapter True 