export AZCOPY_CONCURRENCY_VALUE="AUTO"
export HF_HOME=/mnt/bn/vl-research/workspace/boli01/.cache/huggingface
export HF_TOKEN="HF_Token"
export HF_HUB_ENABLE_HF_TRANSFER="1"

python3 -m pip install --upgrade pip

python3 -m pip install "sglang[all]"

cd /mnt/bn/vl-research/workspace/boli01/projects/sglang/python
python3 -m pip install -e .

pip install hf_transfer

nvidia-smi

python3 -m torch.utils.collect_env

python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer