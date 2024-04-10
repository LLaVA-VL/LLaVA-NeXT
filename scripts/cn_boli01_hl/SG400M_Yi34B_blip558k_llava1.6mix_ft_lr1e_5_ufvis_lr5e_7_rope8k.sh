#!/bin/bash
NAS_REGION="vl-research-cn-boli01-hl";
USER_PROJECT="boli01"
# set up wandb
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=llava-vl
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

export ACCELERATE_DEBUG_MODE="1"
export HF_HOME=/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface
export HF_TOKEN="HF_Token"
export HF_HUB_ENABLE_HF_TRANSFER="1"

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118;
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118;

############### Prepare Envs #################

if [ -n "$USER_PROJECT" ]; then
    cd /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/LLaVA_Next
else
    cd /mnt/bn/${NAS_REGION}/workspace/projects/LLaVA_Next
fi

git config --global --add safe.directory '*'

python3 -m pip install --upgrade pip;

python3 -m pip install -e ".[train]"

python3 -m pip install ninja
python3 -m pip install flash-attn --no-build-isolation

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
export NCCL_DEBUG=WARN


PORT=26000
GPUS="0,1,2,3,4,5,6,7"

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

################ Arnold Jobs ################

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# stage 1
MODEL_VERSION="Nous-Hermes-2-Yi-34B"
PROMPT_VERSION="plain"
BASE_RUN_NAME="dist4_llavanext-${VISION_MODEL_VERSION_CLEAN}-${MODEL_VERSION}-mlp2x_gelu-pretrain_blip558k_${PROMPT_VERSION}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
# torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# llava/train/train_mem.py \
#     --deepspeed scripts/zero3.json \
#     --model_name_or_path "NousResearch/$MODEL_VERSION" \
#     --version $PROMPT_VERSION \
#     --data_path /mnt/bn/${NAS_REGION}/data/llava_data/blip_558k/blip_558k_plain.json \
#     --image_folder /mnt/bn/${NAS_REGION}/data/llava_data/blip_558k/images \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_tunable_parts="mm_mlp_adapter" \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --bf16 True \
#     --run_name $BASE_RUN_NAME \
#     --output_dir /mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --rope_scaling_factor 2 \
#     --rope_scaling_type "linear" \
#     --model_max_length 8192 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 8 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --torch_compile True \
#     --torch_compile_backend "inductor"

# Stage 1.5
PROMPT_VERSION="mistral_direct"
FINAL_RUN_NAME="dist4_llavanext-${VISION_MODEL_VERSION_CLEAN}-${MODEL_VERSION}-pretrain_blip558k_plain-finetune_la1_6mix_lr5e_7_anyres_ufvis_lr1e_6"
echo "RUN_NAME: ${FINAL_RUN_NAME}"
torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "/mnt/bn/${NAS_REGION}/checkpoints/$MODEL_VERSION" \
    --version $PROMPT_VERSION \
    --data_path /mnt/bn/${NAS_REGION}/data/llava_instruct/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_dca45k_synden40k_cococaps20k_sg40kt2k_ori.json \
    --image_folder /mnt/bn/${NAS_REGION}/data/llava_data \
    --pretrain_mm_mlp_adapter="/mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=5e-7 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $FINAL_RUN_NAME \
    --output_dir /mnt/bn/${NAS_REGION}/checkpoints/$FINAL_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --rope_scaling_factor 2 \
    --rope_scaling_type "linear" \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor"

function azcopy_upload() {
    # Assuming the first argument is SRC and the second is TGT
    local SRC="$1"
    local TGT="$2"
    local SAS_TOKEN="?sv=2023-01-03&st=2023-12-23T13%3A48%3A31Z&se=2024-06-30T13%3A48%3A00Z&sr=c&sp=racwdxltf&sig=K77ocq6Ram1uYMenQJZJl%2BBayH%2Bg4e10Raci6wzQY3M%3D"
    # Executing the azcopy command with the provided SRC and TGT
    /mnt/bn/${NAS_REGION}/software/azcopy copy "$SRC" "https://chunyldev.blob.core.windows.net/output/$TGT$SAS_TOKEN" --recursive --overwrite=ifSourceNewer
}

azcopy_upload "/mnt/bn/${NAS_REGION}/checkpoints/$FINAL_RUN_NAME" "projects/llava_data/checkpoints/"

exit 0