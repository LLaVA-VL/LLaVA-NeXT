#!/bin/bash
ROOT_DIR="root to LLaVA-NeXT-Video"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

OPENAIKEY="INPUT YOUR OPENAI API"

SAVE_DIR=$1

python3 llava/eval/evaluate_benchmark_video_detail_description.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR/pred.json \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results.json \
    --num_chunks 1 \
    --num_tasks 16 \
    --api_key $OPENAIKEY \