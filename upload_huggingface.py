from huggingface_hub import upload_folder

# Define your repository name on Hugging Face (it will be created if not existing)
repo_id = "mikarbx/llava-next-mobilenetv2"

# Path to your checkpoint folder
checkpoint_dir = "/home/azureuser/llava-next/checkpoints/llavanext-mikarbx_mobilenetv2-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain-mid/checkpoint-33000"

# Upload the checkpoint folder
upload_folder(
    folder_path=checkpoint_dir,
    repo_id=repo_id,
    commit_message="Initial checkpoint upload",
    ignore_patterns=["global_step*", "*.py", "*.pt", "*.pth", "*.bin", "zero_to_fp32.py", "trainer_state.json"]
)
