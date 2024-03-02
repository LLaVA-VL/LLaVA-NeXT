export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

cd /mnt/bn/vl-research-cn-lf/workspace/projects/ml_envs

bash Miniconda3-latest-Linux-x86_64.sh -b -u;
source ~/miniconda3/etc/profile.d/conda.sh -b -u;

conda init bash;

source ~/.bashrc;

bash /mnt/bn/vl-research-cn-lf/workspace/projects/LLaVA_Next/playground/submit_jobs_sglang.sh &

# Wait for a certain time, e.g., 10 seconds
sleep 600;
echo "Web service initialized";
python /mnt/bn/vl-research-cn-lf/workspace/projects/LLaVA_Next/playground/sgl_llava_inference_multinode.py --image_folder=/mnt/bn/vl-research-cn-lf/data/cc3m/images --result_file /mnt/bn/vl-research-cn-lf/workspace/projects/LLaVA_Next/playground/cc3m_llava34b_cap/cc3m_sglang_cap.json --dist=0 --total_dist=24 --parallel=32 --port=30000