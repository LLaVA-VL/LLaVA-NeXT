#!/bin/bash

# set up wandb
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381

export WANDB_ENTITY=libo0013
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"
NAS_REGION="vl-research"
export ACCELERATE_DEBUG_MODE="1"
export HF_HOME=/mnt/bn/${NAS_REGION}/.cache/huggingface
export HF_TOKEN="HF_Token"
export HF_HUB_ENABLE_HF_TRANSFER="1"

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

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

############### Prepare Envs #################
cd /mnt/bn/${NAS_REGION}/workspace/boli01/projects/LLaVA_Next

git config --global --add safe.directory '*'
python3 -m pip install --upgrade pip;

python3 -m pip install -e ".[train]"

python3 -m pip install git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4
python3 -m pip install ninja
python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

################ Arnold Jobs ################
LLM_VERSION="Qwen/Qwen1.5-0.5B-Chat"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION="qwen_1_5"
PRETRAIN_DATA_VERSION="blip558k"
############### Pretrain ################

PROMPT_VERSION="qwen_1_5"
RATIO=anyres_max_9
BASE_RUN_NAME="llavanext-Qwen_Qwen1.5-0.5B-Chat-google_siglip-so400m-patch14-384-mlp2x_gelu-pretrain_blip558k_plain_bo"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

FINAL_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune_la1_6mix_fvis_direct32k_${RATIO}_haos_baseline"
echo "FINAL_RUN_NAME: ${FINAL_RUN_NAME}"
if [ ! -f "/mnt/bn/${NAS_REGION}/checkpoints/${FINAL_RUN_NAME}/config.json" ];then
    torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
        llava/train/train_mem.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path $LLM_VERSION \
        --version $PROMPT_VERSION \
        --data_path "/mnt/bn/${NAS_REGION}/data/llava_instruct/{llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_dca45k_synden40k_cococaps20k_sg40kt2k_ori_fix_ai2d}.json" \
        --image_folder /mnt/bn/${NAS_REGION}/data/llava_data \
        --pretrain_mm_mlp_adapter="/mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
        --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
        --vision_tower="${VISION_MODEL_VERSION}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --image_aspect_ratio $RATIO \
        --image_grid_pinpoints  "[[384, 384], [384, 768], [384, 1152], [384, 1536], [768, 768], [384, 1920], [768, 768], [384, 2304], [768, 1152], [384, 2688], [768, 1152], [384, 3072], [768, 1536], [384, 3456], [768, 1536], [1152, 1152], [768, 1920], [1152, 1152], [768, 1920], [1152, 1152], [768, 2304], [1152, 1536], [768, 2304], [1152, 1536], [768, 2688], [1152, 1536], [768, 2688], [1152, 1920], [768, 3072], [1152, 1920], [1536, 1536], [768, 3072], [1152, 1920], [1536, 1536], [768, 3456], [1152, 2304], [1536, 1536], [768, 3456], [1152, 2304], [1536, 1536], [768, 3840], [1152, 2304], [1536, 1920], [768, 3840], [1152, 2688], [1536, 1920], [768, 4224], [1152, 2688], [1536, 1920], [768, 4224], [1152, 2688], [1536, 1920], [768, 4608], [1152, 3072], [1536, 2304], [768, 4608], [1152, 3072], [1536, 2304], [1920, 1920], [768, 4992], [1152, 3072], [1536, 2304], [1920, 1920], [768, 4992], [1152, 3456], [1536, 2304], [1920, 1920], [768, 5376], [1152, 3456], [1536, 2688], [1920, 1920], [768, 5376], [1152, 3456], [1536, 2688], [1920, 1920], [768, 5760], [1152, 3840], [1536, 2688], [1920, 2304], [768, 5760], [1152, 3840], [1536, 2688], [1920, 2304], [768, 6144], [1152, 3840], [1536, 3072], [1920, 2304], [768, 6144], [1152, 4224], [1536, 3072], [1920, 2304], [768, 6528], [1152, 4224], [1536, 3072], [1920, 2304], [768, 6528], [1152, 4224], [1536, 3072], [1920, 2688], [768, 6912], [1152, 4608], [1536, 3456], [1920, 2688], [2304, 2304], [768, 6912], [1152, 4608], [1536, 3456], [1920, 2688], [2304, 2304], [1152, 4608], [1536, 3456], [1920, 2688], [2304, 2304], [1152, 4992], [1536, 3456], [1920, 2688], [2304, 2304], [1152, 4992], [1536, 3840], [1920, 3072], [2304, 2304]]" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name $FINAL_RUN_NAME \
        --output_dir /mnt/bn/${NAS_REGION}/checkpoints/$FINAL_RUN_NAME \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --report_to wandb
fi

############################# Eval ################################
export OPENAI_API_KEY="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"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export WANDB_REPORT_API_ENABLE_V2=True
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=libo0013
export WANDB_MODE=online

cd /mnt/bn/vl-research/workspace/boli01/projects/lmms-eval
python3 -m pip install -e .

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained="/mnt/bn/${NAS_REGION}/checkpoints/${FINAL_RUN_NAME}",conv_template="$PROMPT_VERSION" \
    --tasks ai2d,chartqa,docvqa_val,mme,mmmu,mathvista_testmini,infovqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${FINAL_RUN_NAME} \
    --output_path ./logs/ \
    --wandb_args 'project=lmms-eval,job_type=eval,entity=llava-vl'