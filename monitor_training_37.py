#!/usr/bin/env python3

import time
from veesion_data_core.tools import get_training_request

training_name = "llava_cheater-37"

print(f"Monitoring training: {training_name}")
print("Key fix in this version:")
print("- Fixed nested try-catch issue preventing timespan clipping from working")
print("- Should now see 'Clipping timespan for...' messages in logs")
print("- Should proceed past data loading to actual training")
print("=" * 70)

while True:
    try:
        training_request = get_training_request(training_name=training_name)
        print(f"[{time.strftime('%H:%M:%S')}] Status: {training_request.status}")
        
        if training_request.status in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"Training finished with status: {training_request.status}")
            break
            
        if training_request.status == "RUNNING":
            print(f"✅ Training is running! Check logs for timespan clipping messages...")
            
        time.sleep(30)  # Check every 30 seconds
        
    except Exception as e:
        print(f"Error checking training status: {e}")
        time.sleep(30) 