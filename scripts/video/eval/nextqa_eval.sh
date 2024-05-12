#!/bin/zsh
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava_next"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' #2,3,4,5,6,7'
# CUDA_VISIBLE_DEVICES='0'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"
echo $GPULIST

CKPT=$1
CONV_MODE=$2
FRAMES=$3
OVERWRITE=$4
PREDEFINED_CONFIGURE=$5
mm_spatial_pool_stride=$6
MODEL_MAX_LENGTH=${7:-0}

PATCHIFY=False


OPENAIKEY=$OPENAI_API_KEY

if [ "$OVERWRITE" = false ]; then
    if [ "$MODEL_MAX_LENGTH" = 0 ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_overwrite_${OVERWRITE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_overwrite_${OVERWRITE}_length_${MODEL_MAX_LENGTH}
    fi
else
    if [ "$MODEL_MAX_LENGTH" = 0 ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}_length_${MODEL_MAX_LENGTH}
    fi
fi

echo $SAVE_DIR


for IDX in $(seq 1 $((CHUNKS))); do
    echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}"
    set -e
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llavavid/eval/model_nextqa.py \
    --model-path $CKPT \
    --video_dir ${ROOT_DIR}/data/LLaMA-VID-Eval/NextQA/NExTVideo \
    --gt_file ${ROOT_DIR}/data/LLaMA-VID-Eval/NextQA/test_data_nextoe/test.csv \
    --output_dir ${ROOT_DIR}/work_dirs/eval_nextoe/$SAVE_DIR \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --patchify_video_feature ${PATCHIFY} \
    --predefined_configure ${PREDEFINED_CONFIGURE} \
    --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
    --for_get_frames_num $FRAMES \
    --model-max-length ${MODEL_MAX_LENGTH:-0} \
    --conv-mode $CONV_MODE &
done

wait

cat ${ROOT_DIR}/work_dirs/eval_nextoe/$SAVE_DIR/${CHUNKS}* > ${ROOT_DIR}/work_dirs/eval_nextoe/$SAVE_DIR/pred.json

cd /mnt/bn/vl-research/workspace/yhzhang/NExT-OE

python eval_oe.py --result_file ${ROOT_DIR}/work_dirs/eval_nextoe/$SAVE_DIR/pred.json
