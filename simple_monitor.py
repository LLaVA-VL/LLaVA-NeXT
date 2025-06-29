#!/usr/bin/env python3
"""
Simple SSH-based monitoring script for training instances
"""

import paramiko
import argparse
import time
from datetime import datetime

def ssh_command(host, command, key_path=None, username='ubuntu'):
    """Execute a command via SSH"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if key_path:
            ssh.connect(host, username=username, key_filename=key_path)
        else:
            ssh.connect(host, username=username)
        
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        ssh.close()
        
        if error and not output:
            return f"Error: {error}"
        return output
    except Exception as e:
        return f"Connection error: {e}"

def get_logs(host, log_path='/var/log/training.log', lines=100, key_path=None):
    """Get training logs"""
    command = f'tail -n {lines} {log_path}'
    return ssh_command(host, command, key_path)

def run_diagnostics(host, key_path=None):
    """Run diagnostic commands"""
    commands = {
        'nvidia-smi': 'nvidia-smi',
        'gpu_processes': 'nvidia-smi pmon -s um -c 1',
        'training_process': 'ps aux | grep -E "(python|train)" | grep -v grep',
        'system_load': 'uptime',
        'memory': 'free -h'
    }
    
    results = {}
    for name, cmd in commands.items():
        results[name] = ssh_command(host, cmd, key_path)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Monitor training instance via SSH')
    parser.add_argument('host', help='Instance IP address or hostname')
    parser.add_argument('--key', help='SSH private key path')
    parser.add_argument('--user', default='ubuntu', help='SSH username')
    parser.add_argument('--log-path', default='/var/log/training.log', help='Training log path')
    parser.add_argument('--lines', type=int, default=100, help='Number of log lines')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Monitor interval')
    
    args = parser.parse_args()
    
    if args.monitor:
        print(f"Monitoring {args.host} every {args.interval} seconds...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                print(f"\n{'='*60}")
                print(f"Update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                logs = get_logs(args.host, args.log_path, 20, args.key)
                print("Recent logs:")
                print("-" * 40)
                print(logs[-1500:])
                
                gpu_status = ssh_command(args.host, 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits', args.key)
                print(f"\nGPU Status (Util%, Mem Used MB, Mem Total MB, Temp C):")
                print("-" * 40)
                print(gpu_status)
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    elif args.diagnostics:
        print(f"Running diagnostics on {args.host}...")
        results = run_diagnostics(args.host, args.key)
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print("-" * 40)
            print(result)
    
    else:
        print(f"Getting logs from {args.host}...")
        logs = get_logs(args.host, args.log_path, args.lines, args.key)
        print("Training logs:")
        print("=" * 60)
        print(logs)

if __name__ == "__main__":
    main()
