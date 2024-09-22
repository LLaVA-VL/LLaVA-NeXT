#!/bin/bash


# You should complete the path of the following attributes:
PROJECT_ROOT="XXXX"
## This could a yaml file for multiple files or a json file for a single file
DATA_PATH="XXXX"
IMAGE_FOLDER="XXXX"
VIDEO_FOLDER="XXXX"


export PYTHONWARNINGS="ignore"


############### Prepare Envs #################
cd $PROJECT_ROOT
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[train]"

python3 -m pip install ninja
python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi
# 取 worker0 第一个 port
ports=($(echo $METIS_WORKER_0_PORT | tr ',' ' '))
port=${ports[0]}
port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}" | awk -F',' '{print $1}')"

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
export NCCL_DEBUG=WARN

PORT=26000
GPUS="0,1,2,3,4,5,6,7"

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-72B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"


# Stage For video
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llava_next_video-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video"
PREV_STAGE_CHECKPOINT=""
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder ${VIDEO_FOLDER} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 12768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;