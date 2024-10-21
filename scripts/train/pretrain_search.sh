#!/bin/bash

# 超参数的取值范围
train_batch_sizes=(32 64 128)
learning_rates=(5e-4 2e-3 1e-3)

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
LLM_VERSION=/remote-home1/share/models/Qwen/Qwen2.5-0.5B
VISION_MODEL_VERSION="/remote-home1/share/models/vision_encoder/clip-vit-large-patch14-336-openai"

num_train_epochs=1
per_device_train_batch_size=8

PROMPT_VERSION=plain

# 遍历所有的超参数组合
for train_batch_size in "${train_batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do

        # 动态计算 gradient_accumulation_steps
        gradient_accumulation_steps=$((train_batch_size / (per_device_train_batch_size * NUM_GPUS)))

        # 生成独特的运行名称
        BASE_RUN_NAME="llavanext-$(basename "$VISION_MODEL_VERSION")-$(basename "$LLM_VERSION")-mlp2x_gelu-pretrain_blip558k_plain-bs${train_batch_size}-lr${lr}"

        # 检查model_card是否存在
        if [ -f "./checkpoints/projectors/${BASE_RUN_NAME}/README.md" ]; then
            echo "Directory ./checkpoints/projectors/${BASE_RUN_NAME}/README.md already exists, skipping..."
            continue
        fi
        
        echo "Running training with batch size: ${train_batch_size}, learning rate: ${lr}, gradient_accumulation_steps: ${gradient_accumulation_steps}"

        # 启动训练
        ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port=20001 \
            llava/train/train_mem.py \
            --deepspeed scripts/zero2.json \
            --model_name_or_path ${LLM_VERSION} \
            --version ${PROMPT_VERSION} \
            --data_path /remote-home1/share/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
            --image_folder /remote-home1/share/data/LLaVA-Pretrain/images \
            --vision_tower ${VISION_MODEL_VERSION} \
            --mm_tunable_parts="mm_mlp_adapter" \
            --mm_vision_select_layer -2 \
            --mm_projector_type mlp2x_gelu \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --bf16 True \
            --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
            --num_train_epochs ${num_train_epochs} \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --save_steps 50000 \
            --learning_rate ${lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 8192 \
            --gradient_checkpointing False \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb \
            --run_name $BASE_RUN_NAME \
            # --attn_implementation sdpa

        # 等待进程结束
        wait

        echo "Finished training with batch size: ${train_batch_size}, learning rate: ${lr}"
        echo "-----------------------------------------------------------"
    done
done
