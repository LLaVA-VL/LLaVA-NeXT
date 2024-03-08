python3 -m pip install --upgrade pip;

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118;
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118;

export HF_HOME=/mnt/bn/vl-research-boli01-cn/.cache/huggingface;
export HF_TOKEN="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";
export HF_HUB_ENABLE_HF_TRANSFER="1";

cd /mnt/bn/vl-research-boli01-cn/projects/zzz/lmms-eval;
pip install -e .;

cd /mnt/bn/vl-research-boli01-cn/projects/zzz/LLaVA_Next;
pip install -e .;

python3 -m pip install ninja;
python3 -m pip install flash-attn --no-build-isolation;

bash /mnt/bn/vl-research-boli01-cn/projects/zzz/LLaVA_Next/cn_scripts/vicuna/internal0.6m_finetune_llava1.6mix_7b_v0.2_unfreeze.sh


accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained="/mnt/bn/vl-research-boli01-cn/projects/zzz/LLaVA_Next/internal_project_checkpoints/llavanext-lmsys_vicuna-7b-v1.5-clip-vit-large-patch14-336-mlp2x_gelu-pretrain_internal0.6m_vicuna_v1_finetune_llava1.6_datamix_unfreezeVIS_1e" \
    --tasks ok_vqa,textcaps_val,mme_test,mmmu,cmmmu,coco2017_cap_val,vizwiz_vqa_val,ai2d,chartqa,pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix debug \
    --output_path ./logs/  \
    --wandb_args 'project=llava-next-lmms-eval,job_type=eval';