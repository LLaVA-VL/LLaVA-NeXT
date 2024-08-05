#!/bin/bash

# Check if three arguments are passed
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <model_path> <question_path> <base_answer_path> <image_folder> <extra_prompt> <N> <temperature>"
    exit 1
fi

# Assign the command line arguments to variables
model_path=$1
question_path=$2
base_answer_path=$3
image_folder=$4
extra_prompt=$5
N=$6
temperature=$7

# Loop over each chunk/process
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
    # Run the Python program in the background
    CUDA_VISIBLE_DEVICES="$chunk_id" python3 llava/eval/model_vqa.py --model-path "$model_path" --question-file "$question_path" --answers-file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --image-folder "$image_folder" --extra-prompt "$extra_prompt" --temperature "$temperature" &

    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="${base_answer_path}/result.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}.jsonl"
for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}/result_${i}.jsonl"
  cat "$input_file" >> "${base_answer_path}/result.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done