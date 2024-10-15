# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0    # uncomment this lead to error
# export NCCL_DEBUG=INFO

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# NUM_GPUS=1
echo "NUM_GPUS: ${NUM_GPUS}"

LLM_VERSION=/remote-home1/share/models/Qwen/Qwen2.5-3B
VISION_MODEL_VERSION=/remote-home1/share/models/vision_encoder/siglip-so400m-patch14-384

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-$(basename "$VISION_MODEL_VERSION")-$(basename "$LLM_VERSION")-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Determine the suffix based on tunable parts
SUFFIX=""
MM_TUNABLE_PARTS="mm_mlp_adapter,mm_language_model"

[[ $MM_TUNABLE_PARTS == *"mm_vision_tower"* ]] && SUFFIX="${SUFFIX}_vt"
[[ $MM_TUNABLE_PARTS == *"mm_mlp_adapter"* ]] && SUFFIX="${SUFFIX}_mlp"
[[ $MM_TUNABLE_PARTS == *"mm_language_model"* ]] && SUFFIX="${SUFFIX}_lm"

MID_RUN_NAME="llavanext-$(basename "$VISION_MODEL_VERSION")-$(basename "$LLM_VERSION")-mlp2x_gelu-finetune_llava_v1_5_mix665k"
MID_RUN_NAME="${MID_RUN_NAME}-tune${SUFFIX}"
DATA_PATH=/remote-home1/share/data/LLaVA-Instruct-665k/llava_v1_5_mix665k.json
image_folder=/remote-home1/share/data/LLaVA-Instruct-665k/

echo "MID_RUN_NAME: ${MID_RUN_NAME}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port=20001 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path=$DATA_PATH \
    --image_folder $image_folder \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts=$MM_TUNABLE_PARTS \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints "[224]" \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --report_to wandb \
    --run_name $MID_RUN_NAME \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
