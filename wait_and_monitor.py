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


def check_workflow_run_success(run_id):
    """Check if a workflow run has successful setup-instances job"""
    jobs_output = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/runs/{run_id}/jobs"])
    if not jobs_output:
        return False, "Could not get jobs"
    
    try:
        jobs_data = json.loads(jobs_output)
        
        # Check create-aws-resources job
        aws_job = None
        setup_job = None
        
        for job in jobs_data["jobs"]:
            if job["name"] == "create-aws-resources":
                aws_job = job
            elif job["name"] == "setup-instances":
                setup_job = job
        
        # If AWS job failed, get the failure reason
        if aws_job and aws_job["conclusion"] == "failure":
            # Get failure reason from logs
            logs = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/jobs/{aws_job['id']}/logs"])
            if logs and "MaxSpotInstanceCountExceeded" in logs:
                return False, "AWS spot instance limit exceeded"
            elif logs and "Error:" in logs:
                error_lines = [line for line in logs.split('\n') if 'Error:' in line]
                if error_lines:
                    return False, f"AWS error: {error_lines[0].split('Error:')[1].strip()}"
                return False, "AWS resource creation failed"
            return False, "AWS resource creation failed (unknown reason)"
        
        # If AWS job succeeded, check setup-instances job
        if aws_job and aws_job["conclusion"] == "success":
            if setup_job and setup_job["conclusion"] == "success":
                return True, "Success"
            elif setup_job and setup_job["conclusion"] == "failure":
                return False, "Setup-instances job failed"
            elif setup_job and setup_job["conclusion"] == "skipped":
                return False, "Setup-instances job was skipped"
            else:
                return False, f"Setup-instances job status: {setup_job['conclusion'] if setup_job else 'not found'}"
        
        # If we get here, something unexpected happened
        aws_status = aws_job["conclusion"] if aws_job else "not found"
        setup_status = setup_job["conclusion"] if setup_job else "not found"
        return False, f"Unexpected job status - AWS: {aws_status}, Setup: {setup_status}"
        
    except Exception as e:
        return False, f"Error checking jobs: {e}"


def find_workflow_run_for_training(training_name, wait_for_success=False, max_wait_minutes=30):
    """Find the workflow run that matches the training name by searching logs"""
    print(f"🔍 Finding workflow run for training: {training_name}")
    
    if wait_for_success:
        print(f"⏳ Waiting up to {max_wait_minutes} minutes for successful workflow run...")
        max_attempts = max_wait_minutes * 2  # Check every 30 seconds
    else:
        max_attempts = 1
    
    checked_runs = set()  # Track already checked workflow runs
    
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"🔄 Attempt {attempt + 1}/{max_attempts} - checking for new workflow runs...")
        
        # Get recent workflow runs
        gh_command = [
            "gh", "run", "list", 
            "--repo", "veesion-io/terraform-scalable-training",
            "--limit", "50",  # Check more runs to find matches
            "--json", "databaseId,status,conclusion,workflowName,createdAt,displayTitle,headBranch"
        ]
        
        output = run_gh_command(gh_command)
        if not output:
            print("❌ Could not get workflow runs")
            if not wait_for_success:
                return None
            time.sleep(30)
            continue
        
        try:
            runs = json.loads(output)
            matching_runs = []
            new_runs_found = False
            
            # Sort runs by creation time (newest first)
            runs.sort(key=lambda x: x["createdAt"], reverse=True)
            
            for run in runs:
                if run["workflowName"] != "Run scalable training":
                    continue
                    
                run_id = run['databaseId']
                
                # Check if this run contains our training name in the logs
                print(f"🔍 Checking run {run_id} for training name '{training_name}'...")
                
                if check_training_name_in_workflow(run_id, training_name):
                    # Skip if we've already checked this run and it was definitively failed
                    if run_id in checked_runs:
                        continue
                    
                    # Check if this run is successful
                    is_successful, status_msg = check_workflow_run_success(run_id)
                    
                    if is_successful:
                        print(f"✅ Found successful workflow run: {run_id} (created: {run['createdAt']})")
                        return run
                    else:
                        print(f"⚠️  Found matching but failed workflow run: {run_id} (created: {run['createdAt']})")
                        print(f"    Failure reason: {status_msg}")
                        matching_runs.append((run, status_msg))
                        
                        # Only mark as checked if it's definitely failed (not just in progress)
                        if "in progress" not in status_msg.lower() and "still" not in status_msg.lower():
                            checked_runs.add(run_id)
                        new_runs_found = True
            
            if matching_runs and not wait_for_success:
                print("❌ Found matching workflows but all failed:")
                for run, status_msg in matching_runs:
                    print(f"   - {run['databaseId']}: {status_msg}")
                return None
            
            if not matching_runs and not wait_for_success:
                print("❌ No workflow runs found matching the training name")
                return None
            
            if wait_for_success and attempt < max_attempts - 1:
                if new_runs_found:
                    print("⏳ No successful run found yet, waiting 30 seconds for new runs...")
                else:
                    print("⏳ No new runs found, waiting 30 seconds...")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ Error parsing workflow runs: {e}")
            if not wait_for_success:
                return None
            time.sleep(30)
    
    if wait_for_success:
        print(f"⏰ Timeout: No successful workflow run found after {max_wait_minutes} minutes")
    
    return None


def check_training_name_in_workflow(run_id, training_name):
    """Check if the training name appears in the workflow run logs"""
    try:
        # Get jobs for this run
        jobs_output = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/runs/{run_id}/jobs"])
        if not jobs_output:
            return False
        
        jobs_data = json.loads(jobs_output)
        
        # Check launch-training job logs (this is where the training name appears)
        for job in jobs_data["jobs"]:
            if job["name"] == "launch-training":
                # Get logs for this job
                logs = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/jobs/{job['id']}/logs"])
                if logs and training_name in logs:
                    print(f"✅ Found training name '{training_name}' in launch-training job logs")
                    return True
        
        return False
        
    except Exception as e:
        print(f"⚠️  Error checking training name in workflow {run_id}: {e}")
        return False


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
            if job["name"] == "setup-instances" and job["conclusion"] == "success":
                setup_job_id = job["id"]
                break
        
        if not setup_job_id:
            print("❌ No successful setup-instances job found")
            return None
        
        # Get logs
        logs = run_gh_command(["gh", "api", f"repos/veesion-io/terraform-scalable-training/actions/jobs/{setup_job_id}/logs"])
        if not logs:
            print("❌ Could not get job logs")
            return None
        
        # Extract IP
        ip_pattern = r'(\d+\.\d+\.\d+\.\d+)\s+ansible_user=[\w-]+\s+node_name=gpu\d+'
        matches = re.findall(ip_pattern, logs)
        
        if matches:
            print(f"✅ Found instance IP: {matches[0]}")
            return matches[0]
        else:
            print("❌ Could not find instance IP in logs")
            
    except Exception as e:
        print(f"❌ Error getting instance IP: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Find training instance IP')
    parser.add_argument('training_name', help='Training name')
    parser.add_argument('--wait', action='store_true', help='Wait for successful workflow run')
    parser.add_argument('--wait-minutes', type=int, default=30, help='Max minutes to wait for success')
    args = parser.parse_args()
    
    print(f"🔍 Finding instance for training: {args.training_name}")
    
    # Find workflow run for this specific training
    workflow_run = find_workflow_run_for_training(args.training_name, wait_for_success=args.wait, max_wait_minutes=args.wait_minutes)
    if not workflow_run:
        if args.wait:
            print("❌ No successful workflow run found within timeout")
        else:
            print("❌ No matching successful workflow run found")
            print("💡 Use --wait to wait for a successful run, or check AWS spot instance availability")
        return 1
    
    # Get instance IP
    instance_ip = get_instance_ip_from_workflow(workflow_run['databaseId'])
    if not instance_ip:
        print("❌ Could not get instance IP")
        return 1
    
    print(f"🎯 Instance IP: {instance_ip}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 