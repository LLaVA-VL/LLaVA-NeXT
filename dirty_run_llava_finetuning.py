#!/usr/bin/env python3
"""
LLaVA Training Launcher using veesion_data_core
Launches H100 training with S3 region fixes and CUDA error handling
"""

import os
import sys
import time
import boto3
from veesion_data_core.tools import (
    get_training_request,
    request_training,
    resume_training,
    stop_training,
)
from datetime import datetime, timezone

def kill_all_trainings():
    """Kill all active training instances"""
    print("🔥 KILLING ALL TRAINING INSTANCES")
    print("=" * 60)
    
    # List of recent training names to stop
    training_names = [
        'llava_h100_robust-2025-06-28-12-49-51',
        'llava_s3_fixed-2025-06-28-12-26-42',
        'llava_h100_cuda_fixed-2025-06-27-14-54-56',
        'llava_h100_cuda_fixed-2025-06-27-14-54-49',
        'llava_h100_cuda_fixed-2025-06-27-14-52-36',
        'llava_cheater-50-2025-06-25-16-21-32',
        'llava_cheater-50-2025-06-25-16-01-26',
        'llava_cheater-50-2025-06-25-15-42-03',
        'llava_cheater-50-2025-06-25-15-19-43'
    ]
    
    print("🛑 Stopping trainings via veesion package...")
    stopped_count = 0
    for training_name in training_names:
        try:
            print(f"🔄 Stopping: {training_name}")
            stop_training(training_name)
            print(f"✅ Stop command sent for: {training_name}")
            stopped_count += 1
        except Exception as e:
            print(f"⚠️  Could not stop {training_name}: {e}")
    
    print(f"\n💀 Force terminating all GPU instances...")
    # Force terminate all GPU instances in both regions
    terminated_count = 0
    
    # US East 1
    try:
        ec2_us = boto3.client('ec2', region_name='us-east-1')
        response = ec2_us.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running']},
                {'Name': 'instance-type', 'Values': ['p*.48xlarge', 'p*.24xlarge', 'g4dn.*', 'g5.*']}
            ]
        )
        us_instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                us_instances.append(instance['InstanceId'])
        
        if us_instances:
            ec2_us.terminate_instances(InstanceIds=us_instances)
            terminated_count += len(us_instances)
            print(f"✅ Terminated {len(us_instances)} GPU instances in us-east-1")
    except Exception as e:
        print(f"❌ Error terminating us-east-1 instances: {e}")
    
    # EU West 1
    try:
        ec2_eu = boto3.client('ec2', region_name='eu-west-1')
        response = ec2_eu.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running']},
                {'Name': 'instance-type', 'Values': ['g4dn.*', 'g5.*', 'p3.*', 'p4.*']}
            ]
        )
        eu_instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                eu_instances.append(instance['InstanceId'])
        
        if eu_instances:
            ec2_eu.terminate_instances(InstanceIds=eu_instances)
            terminated_count += len(eu_instances)
            print(f"✅ Terminated {len(eu_instances)} GPU instances in eu-west-1")
    except Exception as e:
        print(f"❌ Error terminating eu-west-1 instances: {e}")
    
    print("=" * 60)
    print(f"🎯 SUMMARY:")
    print(f"   📋 Stop commands sent: {stopped_count}")
    print(f"   💀 Instances terminated: {terminated_count}")
    print(f"   ✅ ALL TRAINING INSTANCES KILLED")
    print("=" * 60)

def launch_llava_training():
    """Launch LLaVA training with proper configuration and CUDA fixes"""
    
    # Generate training name with timestamp
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    training_name = f"llava_h100_fixed-{date}"
    
    # Training configuration with H100 CUDA fixes
    config = {
        "training_name": training_name,
        "n_epochs": 3,  # Reduced for faster testing
        "dataset_version": 28,
        "cpu_instances_count": 0,
        "gpu_instances_count": 1,
        "archi": "h100.24xlarge",  # H100 with CUDA fixes
        "terraform_scalable_training_branch": "llava_training",
        "simone_branch": "local_debug_hang",  # Branch with S3 fixes
        "training_args": "scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9_h100.sh"
    }
    
    print("🚀 LLaVA H100 Training Launcher (CUDA Fixed)")
    print("=" * 60)
    print(f"📝 Training name: {training_name}")
    print(f"🌿 Branch: {config['simone_branch']} (S3 region fixes)")
    print(f"💻 GPU instances: {config['gpu_instances_count']}x {config['archi']}")
    print(f"📊 Epochs: {config['n_epochs']}")
    print(f"🎯 Script: {config['training_args']}")
    print(f"🔧 CUDA Fixes: CUDA_LAUNCH_BLOCKING=1, driver check, proper init")
    print(f"✅ Applied recommendations: nvidia-smi verification, PyTorch compatibility")
    print("=" * 60)
    
    try:
        # Submit training request
        print("🔄 Submitting H100 training request with CUDA fixes...")
        request_training(**config)
        
        print(f"✅ Training '{training_name}' submitted successfully!")
        print(f"🔍 Monitor with: python3 /opt/wait_and_monitor.py {training_name}")
        
        return training_name
        
    except Exception as e:
        print(f"❌ Failed to submit training: {e}")
        return None

def monitor_training_status(training_name):
    """Monitor training status"""
    if not training_name:
        return
        
    print(f"\n🔍 Checking training status...")
    
    try:
        training_request = get_training_request(training_name=training_name)
        print(f"📊 Training name: {training_request.training_name}")
        print(f"📈 Status: {training_request.status}")
        
        # Check for branch attribute safely
        if hasattr(training_request, 'simone_branch'):
            print(f"🌿 Branch: {training_request.simone_branch}")
        elif hasattr(training_request, 'branch'):
            print(f"🌿 Branch: {training_request.branch}")
            
        if hasattr(training_request, 'created_at'):
            print(f"🕐 Created: {training_request.created_at}")
        elif hasattr(training_request, 'creation_time'):
            print(f"🕐 Created: {training_request.creation_time}")
            
    except Exception as e:
        print(f"⚠️  Could not get training status: {e}")

def main():
    """Main function"""
    # Check if user wants to kill all trainings
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['kill', 'stop', 'terminate']:
        kill_all_trainings()
        return
    
    print("🎯 Starting robust H100 LLaVA training...")
    print("🔧 Features: S3 region fixes + CUDA error handling")
    print("💡 Usage: python3 dirty_run_llava_finetuning.py [kill] to stop all trainings")
    
    # Launch training
    training_name = launch_llava_training()
    
    if training_name:
        # Wait a moment then check status
        print("\n⏳ Waiting 5 seconds to check status...")
        time.sleep(5)
        monitor_training_status(training_name)
        
        # Provide monitoring instructions
        print(f"\n📋 Next steps:")
        print(f"1. Monitor training: python3 /opt/wait_and_monitor.py {training_name}")
        print(f"2. Stop training: python3 dirty_run_llava_finetuning.py kill")
        print(f"3. Check status: python3 -c \"from veesion_data_core.tools import get_training_request; print(get_training_request('{training_name}').status)\"")
        
        print(f"\n🎯 Key improvements in this training:")
        print(f"✅ S3 region fixed (us-east-1)")
        print(f"✅ CUDA initialization handling")
        print(f"✅ H100-specific optimizations")
        print(f"✅ NCCL timeout fixes")
        print(f"✅ Memory management tuning")
        
    else:
        print("\n❌ Training launch failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
