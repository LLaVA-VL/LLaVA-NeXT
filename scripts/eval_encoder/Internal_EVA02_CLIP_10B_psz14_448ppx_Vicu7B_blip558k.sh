#!/bin/bash

# set up wandb
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=llava-vl
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

# set up llava dev env
cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next

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
# export NCCL_DEBUG=INFO


PORT=26000
GPUS="0,1,2,3,4,5,6,7"

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

MODEL_VERSION="lmsys/vicuna-7b-v1.5"
MODEL_VERSION_CLEAN="${MODEL_VERSION//\//_}"
VISION_MODEL_VERSION="Internal-EVA02-CLIP-10B-14-448"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PRETRAINED="/mnt/bn/vl-research/checkpoints/eva-clip-10b/EVA02-CLIP-10B-14-up448/checkpoints/epoch_50/mp_rank_00_model_states.pt"
PROMPT_VERSION=plain
DATA_VERSION="blip558k"

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

RUN_NAME="llavanext-${MODEL_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain"
echo "RUN_NAME: ${RUN_NAME}"
deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero2.json \
    --model_name_or_path ${MODEL_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/bn/vl-research/data/llava/blip_6m/blip_558k_plain.json \
    --image_folder /mnt/bn/vl-research/data/llava/blip_6m/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./project_checkpoints/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${RUN_NAME}

PROMPT_VERSION="vicuna_v1"
RUN_NAME="llavanext-${MODEL_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-pretrain_${DATA_VERSION}_${PROMPT_VERSION}_finetune_llava1.6_datamix_unfreezeVIS_1e"
echo "RUN_NAME: ${RUN_NAME}"
deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero2.json \
    --model_name_or_path $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_dca45k_synden40k_cococaps20k_sg40kt2k_ori.json \
    --image_folder /mnt/bn/vl-research/data/llava \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./project_checkpoints/llavanext-${MODEL_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(448, 896), (896, 448), (896, 896), (1344, 448), (448, 1344)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./project_checkpoints/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME


PROMPT_VERSION="vicuna_v1"
RUN_NAME="llavanext-${MODEL_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-pretrain_${DATA_VERSION}_${PROMPT_VERSION}_finetune_llava1.6_datamix_freezeVIS_1e"
echo "RUN_NAME: ${RUN_NAME}"

deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero2.json \
    --model_name_or_path ${MODEL_VERSION} \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_dca45k_synden40k_cococaps20k_sg40kt2k_ori.json \
    --image_folder /mnt/bn/vl-research/data/llava \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_MODEL_PRETRAINED} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./project_checkpoints/llavanext-${MODEL_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(448, 896), (896, 448), (896, 896), (1344, 448), (448, 1344)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./project_checkpoints/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME