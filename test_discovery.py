#!/usr/bin/env python3
"""
Test script to verify GitHub Actions workflow discovery and IP extraction
"""

import subprocess
import json
import re
import sys
from datetime import datetime
from typing import Optional, Dict, List


def run_gh_command(command: List[str]) -> Optional[str]:
    """Run a GitHub CLI command and return the output"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"GitHub CLI error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("GitHub CLI command timed out")
        return None
    except Exception as e:
        print(f"Error running GitHub CLI: {e}")
        return None


def find_training_workflow_run(training_name: str) -> Optional[Dict]:
    """Find the GitHub Actions workflow run for the training"""
    print(f"Looking for GitHub Actions workflow run for training '{training_name}'...")
    
    # Extract the timestamp from training name to help find the right run
    training_time = None
    if "2025-" in training_name:
        try:
            timestamp_part = training_name.split("2025-")[1]
            year, month, day, hour, minute, second = "2025", timestamp_part[:2], timestamp_part[3:5], timestamp_part[6:8], timestamp_part[9:11], timestamp_part[12:14]
            training_time = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            print(f"Training submitted at: {training_time}")
        except:
            print("Could not parse training timestamp")
    
    # Get recent workflow runs
    gh_command = [
        "gh", "run", "list", 
        "--repo", "veesion-io/terraform-scalable-training",
        "--limit", "10",
        "--json", "databaseId,status,conclusion,url,workflowName,createdAt,displayTitle"
    ]
    
    output = run_gh_command(gh_command)
    if not output:
        print("Failed to get workflow runs")
        return None
    
    try:
        runs = json.loads(output)
        
        # Look for "Run scalable training" workflows
        for run in runs:
            if run["workflowName"] == "Run scalable training":
                run_time = datetime.fromisoformat(run["createdAt"].replace('Z', '+00:00'))
                
                # If we have training time, check if this run is close to when training was submitted
                if training_time:
                    time_diff = abs((run_time - training_time.replace(tzinfo=run_time.tzinfo)).total_seconds())
                    if time_diff > 300:  # More than 5 minutes difference
                        continue
                
                print(f"Found potential workflow run: {run['databaseId']} (status: {run['status']}, created: {run['createdAt']})")
                
                # If run is completed successfully, return it
                if run["status"] == "completed" and run["conclusion"] == "success":
                    return run
                    
    except json.JSONDecodeError as e:
        print(f"Error parsing workflow runs: {e}")
    
    return None


def get_instance_ip_from_workflow(run_id: str, training_name: str) -> Optional[str]:
    """Extract instance IP from workflow run logs"""
    print(f"Getting instance IP from workflow run {run_id}...")
    
    # First, get the jobs for this run
    gh_command = ["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/runs/{run_id}/jobs"]
    
    output = run_gh_command(gh_command)
    if not output:
        print("Failed to get workflow jobs")
        return None
    
    try:
        jobs_data = json.loads(output)
        
        # Find the setup-instances job
        setup_job_id = None
        for job in jobs_data["jobs"]:
            if job["name"] == "setup-instances" and job["conclusion"] == "success":
                setup_job_id = job["id"]
                break
        
        if not setup_job_id:
            print("Could not find successful setup-instances job")
            return None
        
        print(f"Found setup-instances job: {setup_job_id}")
        
        # Get the logs from the setup-instances job
        gh_command = ["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/jobs/{setup_job_id}/logs"]
        
        logs = run_gh_command(gh_command)
        if not logs:
            print("Failed to get job logs")
            return None
        
        # Extract IP address from logs
        ip_pattern = r'(\d+\.\d+\.\d+\.\d+)\s+ansible_user=[\w-]+\s+node_name=gpu\d+'
        matches = re.findall(ip_pattern, logs)
        
        if matches:
            ip = matches[0]
            print(f"✅ Found instance IP: {ip}")
            return ip
        else:
            print("Could not find instance IP in logs")
            return None
            
    except json.JSONDecodeError as e:
        print(f"Error parsing jobs data: {e}")
        return None


def test_ssh_connection(ip: str):
    """Test SSH connection to the instance"""
    print(f"Testing SSH connection to {ip}...")
    
    ssh_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10', '-i', '/home/veesion/aws/scalable_training_ireland.pem', f'ec2-user@{ip}', 'echo "SSH connection successful"']
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print(f"✅ SSH connection successful: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ SSH connection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ SSH connection error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_discovery.py <training_name>")
        sys.exit(1)
    
    training_name = sys.argv[1]
    
    print(f"🔄 Testing workflow discovery for: {training_name}")
    
    # Step 1: Find the GitHub Actions workflow run
    workflow_run = find_training_workflow_run(training_name)
    
    if not workflow_run:
        print("❌ Could not find GitHub Actions workflow run")
        sys.exit(1)
    
    print(f"✅ Found workflow run: {workflow_run['databaseId']}")
    
    # Step 2: Extract instance IP from workflow logs
    instance_ip = get_instance_ip_from_workflow(str(workflow_run['databaseId']), training_name)
    
    if not instance_ip:
        print("❌ Could not extract instance IP from workflow logs")
        sys.exit(1)
    
    # Step 3: Test SSH connection
    if test_ssh_connection(instance_ip):
        print(f"🎉 All tests passed! Instance IP: {instance_ip}")
    else:
        print("⚠️ Workflow discovery worked but SSH connection failed")


if __name__ == "__main__":
    main() 