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
export HF_TOKEN="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"
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
ACCELERATE_CPU_AFFINITY=1

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen1.5-72B-Chat"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="plain"
BASE_RUN_NAME="llavanext-Qwen_Qwen1.5-72B-Chat-google_siglip-so400m-patch14-384-mlp2x_gelu-pretrain_blip558k_plain_bs256"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/bn/${NAS_REGION}/data/llava_data/blip_558k/blip_558k_plain.json \
    --image_folder /mnt/bn/${NAS_REGION}/data/llava_data/blip_558k/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 24000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
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

azcopy_upload "./project_checkpoints/${FINAL_RUN_NAME}" "projects/llava_data/checkpoints/"

exit 0;
############################# Eval ################################
# cd /mnt/bn/vl-research/workspace/boli01/projects/lmms-eval
# export OPENAI_API_KEY="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"
# export PYTHONWARNINGS=ignore
# export TOKENIZERS_PARALLELISM=false
# export WANDB_REPORT_API_ENABLE_V2=True
# export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
# export WANDB_ENTITY=libo0013
# export WANDB_MODE=online

# accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/project_checkpoints/${MID_RUN_NAME}" \
#     --tasks ai2d,chartqa,docvqa_val,coco2017_cap_val,mme,mmmu_val,textcaps_val,scienceqa_img,vizwiz_vqa_val,pope,ok_vqa,dc100_en,mmvet,llava_in_the_wild \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix ${MID_RUN_NAME} \
#     --output_path ./logs/ \
#     --wandb_args 'project=lmms-eval,job_type=eval,entity=libo0013';