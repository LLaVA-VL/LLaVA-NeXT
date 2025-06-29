#!/usr/bin/env python3
"""
Script to find and monitor GPU training instances on AWS.
This script can:
1. Find running GPU instances
2. Retrieve training logs
3. Run diagnostic commands like nvidia-smi
4. Monitor training progress

Usage examples:
# Auto-detect and monitor latest training instance
python monitor_training_instance.py --auto --monitor

# Find training instances for specific training job
python monitor_training_instance.py --training-name "llava-training" --auto

# Get logs from latest training instance
python monitor_training_instance.py --auto --lines 500

# Run diagnostics on latest training instance  
python monitor_training_instance.py --auto --diagnostics

# Manual selection from found instances
python monitor_training_instance.py --training-name "my-training"

# Get logs from a specific instance
python monitor_training_instance.py --instance-id i-1234567890abcdef0
"""

import boto3
import paramiko
import time
import argparse
import json
import sys
from typing import List, Dict, Optional
from datetime import datetime


class TrainingInstanceMonitor:
    def __init__(self, region='us-east-1', profile=None):
        """Initialize AWS session and clients"""
        self.session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.ec2 = self.session.client('ec2', region_name=region)
        self.ssm = self.session.client('ssm', region_name=region)
        
    def find_training_instances(self, 
                              instance_types: List[str] = None,
                              training_name: str = None,
                              name_pattern: str = None) -> List[Dict]:
        """Find running GPU instances that might be training"""
        
        # Default GPU instance types if none specified
        if instance_types is None:
            instance_types = ['p3.2xlarge', 'p3.8xlarge', 'p3.16xlarge', 'p3dn.24xlarge',
                            'p4d.24xlarge', 'g4dn.xlarge', 'g4dn.2xlarge', 'g4dn.4xlarge',
                            'g4dn.8xlarge', 'g4dn.12xlarge', 'g4dn.16xlarge', 'g5.xlarge',
                            'g5.2xlarge', 'g5.4xlarge', 'g5.8xlarge', 'g5.12xlarge', 'g5.16xlarge']
        
        filters = [
            {'Name': 'instance-state-name', 'Values': ['running']},
            {'Name': 'instance-type', 'Values': instance_types}
        ]
        
        # Add name filter if specified
        if name_pattern:
            filters.append({'Name': 'tag:Name', 'Values': [f'*{name_pattern}*']})
        
        try:
            response = self.ec2.describe_instances(Filters=filters)
            instances = []
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    
                    instance_info = {
                        'InstanceId': instance['InstanceId'],
                        'InstanceType': instance['InstanceType'],
                        'State': instance['State']['Name'],
                        'LaunchTime': instance['LaunchTime'],
                        'PublicIpAddress': instance.get('PublicIpAddress'),
                        'PrivateIpAddress': instance.get('PrivateIpAddress'),
                        'Tags': tags
                    }
                    
                    # Check if this looks like a training instance
                    is_training_instance = False
                    
                    # Look for common training-related tags/names
                    training_indicators = [
                        'training', 'train', 'scalable', 'job', 'worker', 'llava', 'model'
                    ]
                    
                    # Check in Name tag
                    name = tags.get('Name', '').lower()
                    if any(indicator in name for indicator in training_indicators):
                        is_training_instance = True
                    
                    # Check in other common training tags
                    for tag_key in ['Job', 'TrainingJob', 'Project', 'Purpose', 'Role']:
                        if tag_key in tags:
                            tag_value = tags[tag_key].lower()
                            if any(indicator in tag_value for indicator in training_indicators):
                                is_training_instance = True
                                break
                    
                    # If training_name is specified, check if it matches
                    if training_name:
                        training_name_lower = training_name.lower()
                        name_matches = training_name_lower in name
                        tag_matches = any(training_name_lower in str(v).lower() for v in tags.values())
                        
                        if name_matches or tag_matches:
                            is_training_instance = True
                        elif not (name_matches or tag_matches):
                            is_training_instance = False
                    
                    # For instances without clear training indicators, include recent GPU instances
                    if not is_training_instance and not training_name:
                        # Include GPU instances launched in the last 24 hours
                        from datetime import datetime, timezone, timedelta
                        recent_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
                        if instance['LaunchTime'] > recent_threshold:
                            is_training_instance = True
                    
                    if is_training_instance:
                        instances.append(instance_info)
            
            # Sort by launch time (most recent first)
            instances.sort(key=lambda x: x['LaunchTime'], reverse=True)
            
            return instances
            
        except Exception as e:
            print(f"Error finding instances: {e}")
            return []
    
    def find_latest_training_instance(self, training_name: str = None) -> Optional[Dict]:
        """Find the most recent training instance"""
        instances = self.find_training_instances(training_name=training_name)
        
        if not instances:
            return None
        
        # Return the most recent instance
        latest = instances[0]
        print(f"Found latest training instance: {latest['InstanceId']} ({latest['InstanceType']})")
        print(f"  Launch Time: {latest['LaunchTime']}")
        print(f"  Private IP: {latest['PrivateIpAddress']}")
        if 'Name' in latest['Tags']:
            print(f"  Name: {latest['Tags']['Name']}")
        
        return latest
    
    def get_logs_via_ssm(self, instance_id: str, log_path: str = '/var/log/training.log', 
                        lines: int = 100) -> Optional[str]:
        """Get logs using AWS Systems Manager"""
        try:
            command = f'tail -n {lines} {log_path}'
            
            response = self.ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={'commands': [command]}
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for command to complete
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    result = self.ssm.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=instance_id
                    )
                    if result['Status'] in ['Success', 'Failed']:
                        break
                except:
                    pass
                time.sleep(1)
            
            if result['Status'] == 'Success':
                return result['StandardOutputContent']
            else:
                return f"Command failed: {result.get('StandardErrorContent', 'Unknown error')}"
                
        except Exception as e:
            print(f"Error getting logs via SSM: {e}")
            return None
    
    def run_diagnostic_commands(self, instance_id: str) -> Dict[str, str]:
        """Run diagnostic commands on the instance"""
        commands = {
            'nvidia-smi': 'nvidia-smi',
            'gpu_processes': 'nvidia-smi pmon -s um -c 1',
            'system_load': 'uptime',
            'memory_usage': 'free -h',
            'disk_usage': 'df -h',
            'training_process': 'ps aux | grep -E "(python|train)" | grep -v grep',
            'gpu_utilization': 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits',
        }
        
        results = {}
        
        for name, command in commands.items():
            result = self._run_command_ssm(instance_id, command)
            results[name] = result
        
        return results
    
    def _run_command_ssm(self, instance_id: str, command: str) -> str:
        """Run a single command via SSM"""
        try:
            response = self.ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={'commands': [command]}
            )
            
            command_id = response['Command']['CommandId']
            
            # Wait for command to complete
            for _ in range(10):
                try:
                    result = self.ssm.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=instance_id
                    )
                    if result['Status'] in ['Success', 'Failed']:
                        break
                except:
                    pass
                time.sleep(1)
            
            if result['Status'] == 'Success':
                return result['StandardOutputContent']
            else:
                return f"Failed: {result.get('StandardErrorContent', 'Unknown error')}"
                
        except Exception as e:
            return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description='Monitor AWS training instances')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--profile', help='AWS profile to use')
    parser.add_argument('--instance-id', help='Specific instance ID to monitor')
    parser.add_argument('--training-name', help='Training job name to search for')
    parser.add_argument('--auto', action='store_true', help='Automatically select latest training instance')
    parser.add_argument('--tag-name', help='Filter instances by name tag pattern')
    parser.add_argument('--log-path', default='/var/log/training.log', help='Path to training log file')
    parser.add_argument('--lines', type=int, default=200, help='Number of log lines to retrieve')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostic commands')
    parser.add_argument('--monitor', action='store_true', help='Continuously monitor training')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TrainingInstanceMonitor(region=args.region, profile=args.profile)
    
    # Try multiple regions if not specified
    regions_to_try = [args.region, 'eu-west-1', 'eu-central-1', 'us-west-2']
    
    selected_instance = None
    
    # Find instances if not specified
    if not args.instance_id:
        
        if args.auto:
            # Auto-select the latest training instance
            print("Auto-detecting latest training instance...")
            for region in regions_to_try:
                monitor_region = TrainingInstanceMonitor(region=region, profile=args.profile)
                selected_instance = monitor_region.find_latest_training_instance(args.training_name)
                if selected_instance:
                    monitor = monitor_region  # Use the monitor for the region where we found the instance
                    args.instance_id = selected_instance['InstanceId']
                    print(f"Auto-selected instance in region {region}")
                    break
            
            if not selected_instance:
                print("No recent training instances found in any region")
                return
        else:
            # Manual selection
            print("Finding training instances...")
            
            instances = monitor.find_training_instances(training_name=args.training_name, name_pattern=args.tag_name)
            
            if not instances:
                print(f"No training instances found in {args.region}")
                # Try other regions
                for region in ['eu-west-1', 'eu-central-1', 'us-west-2']:
                    if region != args.region:
                        print(f"Trying region {region}...")
                        monitor_region = TrainingInstanceMonitor(region=region, profile=args.profile)
                        instances = monitor_region.find_training_instances(training_name=args.training_name, name_pattern=args.tag_name)
                        if instances:
                            monitor = monitor_region
                            print(f"Found instances in {region}")
                            break
                
                if not instances:
                    print("No training instances found in any region")
                    return
            
            print(f"Found {len(instances)} training instances:")
            for i, instance in enumerate(instances):
                print(f"{i+1}. {instance['InstanceId']} ({instance['InstanceType']}) - {instance['PrivateIpAddress']}")
                if 'Name' in instance['Tags']:
                    print(f"   Name: {instance['Tags']['Name']}")
                print(f"   Launch Time: {instance['LaunchTime']}")
                print()
            
            if len(instances) == 1:
                selected_instance = instances[0]
                print("Auto-selecting the only instance found.")
            else:
                choice = input(f"Select instance (1-{len(instances)}) or 'auto' for latest: ")
                if choice.lower() == 'auto':
                    selected_instance = instances[0]  # Already sorted by launch time
                    print("Auto-selected latest instance.")
                else:
                    try:
                        selected_instance = instances[int(choice) - 1]
                    except (ValueError, IndexError):
                        print("Invalid selection")
                        return
            
            args.instance_id = selected_instance['InstanceId']
    
    print(f"Connecting to instance: {args.instance_id}")
    
    if args.monitor:
        # Continuous monitoring
        print(f"Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{'='*60}")
                print(f"Update at {timestamp}")
                print(f"{'='*60}")
                
                # Get recent logs
                logs = monitor.get_logs_via_ssm(args.instance_id, args.log_path, lines=20)
                
                if logs:
                    print("Recent training logs:")
                    print("-" * 40)
                    print(logs[-1500:])  # Last 1500 characters
                
                # Get GPU status
                gpu_status = monitor._run_command_ssm(args.instance_id, 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits')
                
                print(f"\nGPU Status:")
                print("-" * 40)
                print(f"GPU Util%, Memory Used MB, Memory Total MB, Temp C:")
                print(gpu_status)
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            
    elif args.diagnostics:
        # Run diagnostics
        print("Running diagnostic commands...")
        results = monitor.run_diagnostic_commands(args.instance_id)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print("-" * 40)
            print(result)
    else:
        # Get logs
        print(f"Retrieving last {args.lines} lines from {args.log_path}...")
        
        logs = monitor.get_logs_via_ssm(args.instance_id, args.log_path, args.lines)
        
        if logs:
            print("Training logs:")
            print("=" * 60)
            print(logs)
        else:
            print("Failed to retrieve logs")


if __name__ == "__main__":
    main()
