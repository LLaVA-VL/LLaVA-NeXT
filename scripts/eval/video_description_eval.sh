#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-next-video"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

OPENAIKEY="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"

SAVE_DIR=$1

python3 llava/eval/evaluate_benchmark_video_detail_description.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR/pred.json \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results.json \
    --num_chunks 1 \
    --num_tasks 16 \
    --api_key $OPENAIKEY \