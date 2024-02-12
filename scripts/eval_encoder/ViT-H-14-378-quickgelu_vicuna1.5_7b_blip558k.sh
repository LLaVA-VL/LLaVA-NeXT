#!/bin/bash

# set up wandb
export WANDB_API_KEY=e464cc107357c7b38e87f239bc3eb2ce5fb73c7c
export WANDB_PROJECT=llava

# set up llava dev env
cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next

if ! pip show llava | grep "Version" | grep "1.7.0.dev0"; then
    python3 -m pip install --upgrade pip  # enable PEP 660 support
    python3 -m pip install -e .

    python3 -m pip install ninja
    python3 -m pip install flash-attn --no-build-isolation
fi

nvidia-smi

# run experiment

# PORT=26000
# GPUS="0,1,2,3,4,5,6,7"

# CUR_DIR=$(cd $(dirname $0); pwd)
# cd $CUR_DIR

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
# export NCCL_DEBUG=INFO

PORT=26000
GPUS="0,1,2,3,4,5,6,7"

MODEL_VERSION="vicuna-7b-v1-5"
VISION_MODEL_VERSION="open_clip_hub:ViT-H-14-378-quickgelu"
VISION_PRETRAINED="dfn5b"

PROMPT_VERSION=plain
DATA_VERSION="blip558k"

deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero3.json \
    --model_name_or_path ./checkpoints/${MODEL_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/bn/vl-research/data/llava/blip_6m/blip_558k_plain.json \
    --image_folder /mnt/bn/vl-research/data/llava/blip_6m/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_PRETRAINED} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llavanext-${MODEL_VERSION}-${VISION_MODEL_VERSION}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb


PROMPT_VERSION="vicuna_v1"
deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct/llava_158k_detailv3reinst_sgpt4v_coco1k_lcs15k_laion8k_wild15k_vqav2_83k_okvqa_9k_aokvqa_17k_mc_ar_refcoco30k_rec_s10_vg86k_reg_f20_gqa72k_ocrvqa80k_docvqa10k_sg40k_ori_p3.json \
    --image_folder /mnt/bn/vl-research/data/llava \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_PRETRAINED} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavanext-${MODEL_VERSION}-${VISION_MODEL_VERSION}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower 2e-6 \
    --bf16 True \
    --output_dir ./checkpoints/llavanext-${MODEL_VERSION}-${VISION_MODEL_VERSION}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain_finetune_llava1.6_datamix_unfreezeVIS_1e \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

PROMPT_VERSION="vicuna_v1"
deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed chunyl_scripts/vc/train/ds_zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct/llava_158k_detailv3reinst_sgpt4v_coco1k_lcs15k_laion8k_wild15k_vqav2_83k_okvqa_9k_aokvqa_17k_mc_ar_refcoco30k_rec_s10_vg86k_reg_f20_gqa72k_ocrvqa80k_docvqa10k_sg40k_ori_p3.json \
    --image_folder /mnt/bn/vl-research/data/llava \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_pretrained ${VISION_PRETRAINED} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavanext-${MODEL_VERSION}-${VISION_MODEL_VERSION}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavanext-${MODEL_VERSION}-${VISION_MODEL_VERSION}-mlp2x_gelu-pretrain_${DATA_VERSION}_plain_finetune_llava1.6_datamix_freezeVIS_1e \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
