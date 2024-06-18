NAS_REGION=vl-research
USER_PROJECT=boli01
export AZCOPY_CONCURRENCY_VALUE="AUTO"
export HF_HOME=/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER="1"

cp /mnt/bn/vl-research/workspace/fengli/.bashrc /home/tiger/
source /home/tiger/.bashrc

conda activate /mnt/bn/vl-research/workspace/boli01/conda_envs/sglang

conda -V

# cd /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/ml_envs

# bash Miniconda3-latest-Linux-x86_64.sh -b -u;
# source ~/miniconda3/etc/profile.d/conda.sh -b -u;

# conda init bash;
# source ~/.bashrc;

# cd /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next;
# python3 -m pip install e ".[train]"

# python3 -m pip install vllm==0.3.3
# python3 -m pip install "sglang[all]"==0.1.13
# python3 -m pip install hf_transfer
# python3 -m pip install outlines==0.0.34
# python3 -m pip install httpx==0.23.3

nvidia-smi
python3 -m torch.utils.collect_env

which python3

python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port=30000 --tp-size=8 &

sleep 300;
echo "Web service initialized";
python3 /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/playground/sgl_llava_inference_multinode.py \
    --dist=${1} --total_dist=6 --parallel=32 --port=30000 \
    --result_file=/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/playground/cc3m_result_file.json
