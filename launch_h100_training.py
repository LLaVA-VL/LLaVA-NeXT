#!/usr/bin/env python3
"""
Launch H100 LLaVA training with CUDA error fixes
"""

import time
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🚀 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def launch_training():
    """Launch the H100 training with fixed CUDA configuration"""
    
    # Training parameters
    training_name = f"llava_h100_fixed_{int(time.time())}"
    script_path = "scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9_h100.sh"
    branch = "local_debug_hang"
    gpu_instances = 1
    instance_type = "h100.24xlarge"
    region = "us-east-1"
    
    print(f"🎯 Launching training: {training_name}")
    print(f"📝 Script: {script_path}")
    print(f"🌿 Branch: {branch}")
    print(f"💻 Instance: {gpu_instances}x {instance_type}")
    print(f"🌍 Region: {region}")
    print("=" * 60)
    
    # Activate environment and launch training
    launch_cmd = f"""
    source env/bin/activate && python3 -c "
import veesion_data_core.tools as tools
try:
    # Use the GitHub Actions workflow to launch training
    print('🚀 Starting training with GitHub Actions...')
    
    # Create a training request using the workflow system
    import requests
    import os
    import json
    
    # Get GitHub token
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print('❌ GITHUB_TOKEN not found in environment')
        exit(1)
    
    # Trigger workflow
    url = 'https://api.github.com/repos/veesion-io/terraform-scalable-training/actions/workflows/training.yml/dispatches'
    headers = {{
        'Authorization': f'token {{github_token}}',
        'Accept': 'application/vnd.github.v3+json'
    }}
    
    data = {{
        'ref': 'main',
        'inputs': {{
            'training_name': '{training_name}',
            'training_script': '{script_path}',
            'branch': '{branch}',
            'gpu_instances_count': '{gpu_instances}',
            'instance_type': '{instance_type}',
            'region': '{region}'
        }}
    }}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 204:
        print('✅ Training workflow triggered successfully!')
        print(f'Training name: {training_name}')
    else:
        print(f'❌ Failed to trigger workflow: {{response.status_code}} - {{response.text}}')
        exit(1)
        
except Exception as e:
    print(f'❌ Error launching training: {{e}}')
    exit(1)
"
    """
    
    if run_command(launch_cmd, "Launching H100 training"):
        print(f"✅ Training {training_name} launched successfully!")
        print(f"🔍 Use this command to monitor: python3 /opt/wait_and_monitor.py {training_name}")
        return training_name
    else:
        print("❌ Failed to launch training")
        return None

def monitor_training(training_name):
    """Monitor the training progress"""
    if not training_name:
        return
        
    print(f"🔍 Starting monitoring for: {training_name}")
    
    monitor_cmd = f"python3 /opt/wait_and_monitor.py {training_name}"
    
    try:
        subprocess.run(monitor_cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")

if __name__ == "__main__":
    print("🚀 H100 LLaVA Training Launcher")
    print("🔧 With CUDA Error Fixes for H100 GPUs")
    print("=" * 50)
    
    # Launch training
    training_name = launch_training()
    
    if training_name:
        # Wait a bit for the workflow to start
        print("⏳ Waiting 30 seconds for workflow to initialize...")
        time.sleep(30)
        
        # Start monitoring
        monitor_training(training_name)
    else:
        print("❌ Training launch failed")
        sys.exit(1) 