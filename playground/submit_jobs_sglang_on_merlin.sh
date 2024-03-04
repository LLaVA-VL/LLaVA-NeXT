export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

cd /mnt/bn/vl-research-cn-lf/workspace/projects/ml_envs

bash Miniconda3-latest-Linux-x86_64.sh -b -u;
source ~/miniconda3/etc/profile.d/conda.sh -b -u;

conda init bash;

source ~/.bashrc;

#!/bin/bash
export AZCOPY_CONCURRENCY_VALUE="AUTO"
export HF_HOME=/mnt/bn/vl-research-cn-lf/workspace/.cache/huggingface
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER="1"

cd /mnt/bn/vl-research-cn-lf/workspace/projects/sglang
/home/tiger/miniconda3/bin/python3 -m pip install --upgrade pip
/home/tiger/miniconda3/bin/python3 -m pip install -e "python[all]"

/home/tiger/miniconda3/bin/python3 -m pip install git+https://github.com/Luodian/vllm.git
/home/tiger/miniconda3/bin/python3 -m pip install hf_transfer

nvidia-smi
# python3 -m torch.utils.collect_env

which python3

cd /mnt/bn/vl-research-cn-lf/workspace/projects/sglang
/home/tiger/miniconda3/bin/python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port=30000 --tp-size=8