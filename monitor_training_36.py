#!/usr/bin/env python3

import time
from veesion_data_core.tools import get_training_request

training_name = "llava_cheater-36"

print(f"Monitoring training: {training_name}")
print("Key fixes in this version:")
print("- Fixed timespan validation error parsing for different error formats")
print("- Added S3 credentials retry logic")
print("- Proper clipping of invalid timespans")
print("=" * 60)

while True:
    try:
        training_request = get_training_request(training_name=training_name)
        print(f"[{time.strftime('%H:%M:%S')}] Status: {training_request.status}")
        
        if training_request.status in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"Training finished with status: {training_request.status}")
            break
            
        if training_request.status == "RUNNING":
            print(f"✅ Training is running! The fixes should be working...")
            
        time.sleep(30)  # Check every 30 seconds
        
    except Exception as e:
        print(f"Error checking training status: {e}")
        time.sleep(30) 