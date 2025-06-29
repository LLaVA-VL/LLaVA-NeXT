#!/usr/bin/env python3
import boto3
import argparse

def find_all_running_instances(region='us-east-1'):
    ec2 = boto3.client('ec2', region_name=region)
    
    try:
        # Find all running instances
        response = ec2.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_info = {
                    'InstanceId': instance['InstanceId'],
                    'InstanceType': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'LaunchTime': instance['LaunchTime'],
                    'PublicIpAddress': instance.get('PublicIpAddress'),
                    'PrivateIpAddress': instance.get('PrivateIpAddress'),
                    'Tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                }
                instances.append(instance_info)
        
        print(f"Found {len(instances)} running instances in {region}:")
        for instance in instances:
            print(f"  {instance['InstanceId']} ({instance['InstanceType']}) - {instance.get('PrivateIpAddress', 'No IP')}")
            if 'Name' in instance['Tags']:
                print(f"    Name: {instance['Tags']['Name']}")
            print(f"    Launch Time: {instance['LaunchTime']}")
            print()
        
        return instances
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    find_all_running_instances(args.region)
