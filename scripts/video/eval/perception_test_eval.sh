#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/LLaVA_dev"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 # '0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=8 #${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

CKPT=$1
CONV_MODE=$2
FRAMES=$3
PATCHIFY=$4
PREDEFINED_CONFIGURE=$5
mm_spatial_pool_stride=$6
OVERWRITE=True

OPENAIKEY=$OPENAI_API_KEY

SAVE_DIR=$(basename $CKPT)__${CONV_MODE}_pathify_${PATCHIFY}_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}
echo $SAVE_DIR

for IDX in $(seq 1 $((CHUNKS))); do
    echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llavavid/eval/model_perception_test_qa.py \
    --model-path $CKPT \
    --video_dir /mnt/bn/vl-research/data/LLaMA-VID/data/LLaMA-VID-Eval/perception_test/test_videos \
    --gt_file_question /mnt/bn/vl-research/data/LLaMA-VID/data/LLaMA-VID-Eval/perception_test/mc_question_test.json \
    --output_dir ./work_dirs/eval_perception/$SAVE_DIR \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --patchify_video_feature ${PATCHIFY} \
    --predefined_configure ${PREDEFINED_CONFIGURE} \
    --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE #&

done

# wait

# python llavavid/eval/eval_activitynet_qa.py \
#     --pred_path ./work_dirs/eval_activitynet/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_activitynet/$SAVE_DIR/results \
#     --output_json ./work_dirs/eval_activitynet/$SAVE_DIR/results.json \
#     --num_chunks $CHUNKS \
#     --api_key $OPENAIKEY \
#     --num_tasks 16 \
#     # --api_key $OPENAIKEY \