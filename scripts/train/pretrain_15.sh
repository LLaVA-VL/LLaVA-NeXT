# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0    # uncomment this lead to error
# export NCCL_DEBUG=INFO

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# NUM_GPUS=1

LLM_VERSION=/remote-home1/share/models/Qwen/Qwen2.5-3B
VISION_MODEL_VERSION="/remote-home1/share/models/vision_encoder/clip-vit-large-patch14-336-openai"

WANDB_PROJECT=Llava-NeXT
############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-$(basename "$VISION_MODEL_VERSION")-$(basename "$LLM_VERSION")-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port=20001 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero0.json \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
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

# You can delete the sdpa attn_implementation if you want to use flash attn