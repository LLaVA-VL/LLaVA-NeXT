#!/usr/bin/env python3
"""
Quick training instance discovery and monitoring
"""

import time
import subprocess
import sys
import json
import re
from datetime import datetime, timedelta
import argparse


def run_gh_command(command, timeout=30):
    """Run a GitHub CLI command quickly"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


def find_latest_workflow_run():
    """Find the most recent successful workflow run with setup-instances"""
    print("🔍 Finding latest successful workflow run...")
    
    gh_command = [
        "gh", "run", "list", 
        "--repo", "veesion-io/terraform-scalable-training",
        "--limit", "20",
        "--json", "databaseId,status,conclusion,workflowName,createdAt"
    ]
    
    output = run_gh_command(gh_command)
    if not output:
        return None
    
    try:
        runs = json.loads(output)
        for run in runs:
            if run["workflowName"] == "Run scalable training":
                # Check if this run has a successful setup-instances job
                jobs_output = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/runs/{run['databaseId']}/jobs"])
                if jobs_output:
                    try:
                        jobs_data = json.loads(jobs_output)
                        for job in jobs_data["jobs"]:
                            if job["name"] == "setup-instances" and job["conclusion"] == "success":
                                print(f"✅ Found workflow run: {run['databaseId']} (status: {run['status']}) with successful setup-instances")
                                return run
                    except:
                        continue
    except:
        pass
    
    return None


def get_instance_ip_from_workflow(run_id):
    """Extract instance IP from workflow run logs"""
    print(f"🔍 Getting instance IP from run {run_id}...")
    
    # Get jobs
    jobs_output = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/runs/{run_id}/jobs"])
    if not jobs_output:
        return None
    
    try:
        jobs_data = json.loads(jobs_output)
        setup_job_id = None
        
        for job in jobs_data["jobs"]:
            if job["name"] == "setup-instances":
                setup_job_id = job["id"]
                break
        
        if not setup_job_id:
            return None
        
        # Get logs
        logs = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/jobs/{setup_job_id}/logs"])
        if not logs:
            return None
        
        # Extract IP
        ip_pattern = r'(\d+\.\d+\.\d+\.\d+)\s+ansible_user=[\w-]+\s+node_name=gpu\d+'
        matches = re.findall(ip_pattern, logs)
        
        if matches:
            print(f"✅ Found instance IP: {matches[0]}")
            return matches[0]
            
    except:
        pass
    
    return None


def ssh_command(host, command):
    """Run SSH command quickly"""
    ssh_cmd = [
        'ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=5',
        '-i', '/home/veesion/aws/scalable_training_ireland.pem',
        f'ec2-user@{host}', command
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        return result.stdout if result.returncode == 0 else None
    except:
        return None


def check_training_status(instance_ip):
    """Quick training status check"""
    print(f"\n🔍 Checking training status on {instance_ip}...")
    
    # Check if training is running
    processes = ssh_command(instance_ip, "ps aux | grep train_mem.py | grep -v grep | wc -l")
    if processes and processes.strip() == "0":
        print("⚠️  No training process found")
        return False
    else:
        print("✅ Training process is running")
    
    # Get recent logs
    log_locations = [
        "/var/log/veesion/deep-tmux.log",
        "/var/log/training.log",
        "/tmp/training.log"
    ]
    
    for log_path in log_locations:
        logs = ssh_command(instance_ip, f"tail -n 10 {log_path} 2>/dev/null")
        if logs and len(logs.strip()) > 0:
            print(f"\n📋 Recent logs from {log_path}:")
            print("-" * 60)
            print(logs[-800:])  # Last 800 chars
            break
    
    # Check for our torch_compile fix
    torch_fix_check = ssh_command(instance_ip, "grep -n 'torch_compile disabled' /var/log/veesion/deep-tmux.log 2>/dev/null | tail -1")
    if torch_fix_check:
        print(f"\n🔧 Torch compile fix detected: {torch_fix_check.strip()}")
    
    # Check for forward/backward pass
    forward_pass = ssh_command(instance_ip, "grep -c 'forward\\|backward\\|step.*loss' /var/log/veesion/deep-tmux.log 2>/dev/null")
    if forward_pass and forward_pass.strip() != "0":
        print(f"🚀 Forward/backward passes detected: {forward_pass.strip()} occurrences")
    else:
        print("⚠️  No forward/backward passes detected yet")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Quick training monitor')
    parser.add_argument('training_name', help='Training name')
    parser.add_argument('--no-monitor', action='store_true', help='Just get IP, no monitoring')
    args = parser.parse_args()
    
    print(f"🚀 Quick monitor for: {args.training_name}")
    
    # Find workflow run
    workflow_run = find_latest_workflow_run()
    if not workflow_run:
        print("❌ No workflow run found")
        return 1
    
    # Get instance IP
    instance_ip = get_instance_ip_from_workflow(workflow_run['databaseId'])
    if not instance_ip:
        print("❌ Could not get instance IP")
        return 1
    
    print(f"🎯 Instance IP: {instance_ip}")
    
    if args.no_monitor:
        print("✅ Discovery complete!")
        return 0
    
    # Monitor training
    try:
        while True:
            if not check_training_status(instance_ip):
                break
            print(f"\n⏳ Waiting 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
