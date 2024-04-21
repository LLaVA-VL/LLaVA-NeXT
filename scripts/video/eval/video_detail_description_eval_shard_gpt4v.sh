#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false


OPENAIKEY=$OPENAI_API_KEY

CKPT=$1
FRAMES=$2
CHUNKS=${3:-1}

SAVE_DIR=$(basename $CKPT)_frames_${FRAMES}


# for IDX in $(seq 1 $CHUNKS); do
#     python3 llavavid/eval/model_video_detail_description.py \
#         --model-path $CKPT \
#         --video_dir ./data/llava_video/video-chatgpt/evaluation/Test_Videos/ \
#         --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR \
#         --output_name pred \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
#         --api_key $OPENAIKEY \
#         --for_get_frames_num $FRAMES &
# done

# wait

python3 llavavid/eval/evaluate_benchmark_video_detail_description.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

