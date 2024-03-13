#!/bin/bash
NAS_REGION="vl-research-boli01-cn"

# set up wandb
export WANDB_API_KEY=e9f0dc0578376e9ce4e1303ae0346da601810f90
export WANDB_ENTITY=kcz358
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

export ACCELERATE_DEBUG_MODE="1"
export HF_HOME=/mnt/bn/${NAS_REGION}/.cache/huggingface
export HF_TOKEN="HF_Token"
export HF_HUB_ENABLE_HF_TRANSFER="1"
# set up llava dev env
cd /mnt/bn/${NAS_REGION}/workspace/zzz/projects/zzz/LLaVA_Next
# git pull
# git checkout dev_decoder

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

wandb login e9f0dc0578376e9ce4e1303ae0346da601810f90
wandb online

################ Arnold Jobs ################

LLM_VERSION="lmsys/vicuna-7b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"

############### Pretrain ################

BASE_RUN_NAME="ds_llava-vicuna-7b-v1-5-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
# torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero2.json \
#     --model_name_or_path ${LLM_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --data_path /mnt/bn/vl-research-cn-lf/data/llava_data/blip_558k/blip_558k_plain.json \
#     --image_folder /mnt/bn/vl-research-cn-lf/data/llava_data/blip_558k/images \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./project_checkpoints/${BASE_RUN_NAME} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "no" \
#     --save_steps 24000 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 16 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name $BASE_RUN_NAME

python3 -m pip install transformers --upgrade

# Stage 1.5
# Experiment go in one stage
PROMPT_VERSION="vicuna_v1"
MID_RUN_NAME="dist1_llavanext-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_one_stage"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path /mnt/bn/${NAS_REGION}/workspace/zzz/projects/zzz/LLaVA_Next/llava_instruct_json/combined_staged_instruct.json \
    --image_folder /mnt/bn/${NAS_REGION}/data/llava_data \
    --pretrain_mm_mlp_adapter="/mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./project_checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --rope_scaling_factor 2 \
    --rope_scaling_type "linear" \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

alias azcopy="/mnt/bn/${NAS_REGION}/software/azcopy"

function azcopy_upload() {
    # Assuming the first argument is SRC and the second is TGT
    local SRC="$1"
    local TGT="$2"
    local SAS_TOKEN="?sv=2023-01-03&st=2023-12-23T13%3A48%3A31Z&se=2024-06-30T13%3A48%3A00Z&sr=c&sp=racwdxltf&sig=K77ocq6Ram1uYMenQJZJl%2BBayH%2Bg4e10Raci6wzQY3M%3D"
    # Executing the azcopy command with the provided SRC and TGT
    /mnt/bn/${NAS_REGION}/software/azcopy copy "$SRC" "https://chunyldev.blob.core.windows.net/output/$TGT$SAS_TOKEN" --recursive --overwrite=ifSourceNewer
}

azcopy_upload "./project_checkpoints/${MID_RUN_NAME}" "projects/llava_data/checkpoints/"

python3 -m pip install transformers==4.37.2

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained="./project_checkpoints/${MID_RUN_NAME}" \
    --tasks ai2d,chartqa,docvqa_val,coco2017_cap_val,mme,mmmu_val,textcaps_val,scienceqa_img,vizwiz_vqa_val,pope,ok_vqa\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix one_stage \
    --output_path ./logs/ \
    --wandb_args 'project=llava_next_one_stage,job_type=eval';