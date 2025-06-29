#!/usr/bin/env python3
"""Simplified video loading test to identify S3 and torchcodec bottlenecks."""

import time
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import torch
import io

def test_s3_download_speed():
    """Test S3 download speed directly."""
    print("🔍 Testing S3 download speed...")
    
    try:
        s3 = boto3.client('s3', region_name='eu-west-1')
        bucket = "scalable-training-dataset"
        prefix = "Simone_28_full/videos/"
        
        # List some videos first
        print("📋 Listing videos...")
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=5
        )
        
        if 'Contents' not in response or len(response['Contents']) == 0:
            print("❌ No videos found in S3")
            return
        
        # Test download of first video
        first_video = response['Contents'][0]
        video_key = first_video['Key']
        video_size = first_video['Size']
        video_id = video_key.replace(prefix, '')
        
        print(f"📹 Testing: {video_id}")
        print(f"📊 Size: {video_size / (1024*1024):.1f} MB")
        
        # Test different download methods
        # 1. Head object (metadata only)
        start = time.time()
        s3.head_object(Bucket=bucket, Key=video_key)
        head_time = time.time() - start
        print(f"⏱️  Head object: {head_time:.3f}s")
        
        # 2. Download first 1MB
        start = time.time()
        response = s3.get_object(Bucket=bucket, Key=video_key, Range='bytes=0-1048575')  # First 1MB
        chunk_data = response['Body'].read()
        chunk_time = time.time() - start
        print(f"⏱️  First 1MB download: {chunk_time:.3f}s ({len(chunk_data) / (1024*1024):.1f} MB)")
        
        # 3. Full download (if small enough)
        if video_size < 50 * 1024 * 1024:  # Less than 50MB
            start = time.time()
            response = s3.get_object(Bucket=bucket, Key=video_key)
            full_data = response['Body'].read()
            full_time = time.time() - start
            download_speed = (video_size / (1024*1024)) / full_time
            print(f"⏱️  Full download: {full_time:.3f}s ({download_speed:.1f} MB/s)")
            
            # Test video decoding if we have torchcodec
            try:
                print("🎬 Testing video decoding...")
                import torchcodec
                from torchcodec.decoders import VideoDecoder
                
                # Create file-like object
                video_file = io.BytesIO(full_data)
                
                decode_start = time.time()
                decoder = VideoDecoder(video_file, device='cpu')
                
                # Try to decode a few frames
                try:
                    frames = decoder.get_frames_played_in_range(1.0, 5.0)  # 4 seconds
                    decode_time = time.time() - decode_start
                    print(f"⏱️  Decode 4s segment: {decode_time:.3f}s ({len(frames.data)} frames)")
                    print(f"📐 Frame shape: {frames.data.shape}")
                    
                    # Test frame sampling
                    sample_start = time.time()
                    num_frames = 40
                    total_frames = len(frames.data)
                    if total_frames > num_frames:
                        interval = (total_frames - 1) / (num_frames - 1)
                        indices = [round(i * interval) for i in range(num_frames)]
                        sampled_frames = frames.data[indices]
                    else:
                        sampled_frames = frames.data
                    sample_time = time.time() - sample_start
                    print(f"⏱️  Frame sampling: {sample_time:.3f}s ({len(sampled_frames)} frames)")
                    
                except Exception as decode_error:
                    print(f"❌ Decode error: {decode_error}")
                    
            except ImportError:
                print("⚠️  torchcodec not available, skipping decode test")
            except Exception as e:
                print(f"❌ Decode test failed: {e}")
        else:
            print(f"⚠️  Video too large ({video_size / (1024*1024):.1f} MB), skipping full download")
            
        return video_id, video_size, head_time, chunk_time
        
    except Exception as e:
        print(f"❌ S3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_s3_connection_settings():
    """Test different S3 connection configurations."""
    print("\n🔧 Testing S3 connection settings...")
    
    bucket = "scalable-training-dataset"
    prefix = "Simone_28_full/videos/"
    
    configs = [
        ("Default", {}),
        ("Retries=1", {"retries": {"max_attempts": 1}}),
        ("Retries=3", {"retries": {"max_attempts": 3}}),
        ("TCP Keepalive", {"tcp_keepalive": True}),
        ("Max Bandwidth", {"max_bandwidth": 100 * 1024 * 1024}),  # 100 MB/s
    ]
    
    for name, config in configs:
        try:
            print(f"\n🧪 Testing {name}...")
            
            if config:
                s3_config = Config(**config)
                s3 = boto3.client('s3', region_name='eu-west-1', config=s3_config)
            else:
                s3 = boto3.client('s3', region_name='eu-west-1')
            
            # Quick head object test
            start = time.time()
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            list_time = time.time() - start
            
            if 'Contents' in response and len(response['Contents']) > 0:
                key = response['Contents'][0]['Key']
                start = time.time()
                s3.head_object(Bucket=bucket, Key=key)
                head_time = time.time() - start
                print(f"  ⏱️  List: {list_time:.3f}s, Head: {head_time:.3f}s")
            else:
                print(f"  ❌ No objects found")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def analyze_training_logs():
    """Check what the training logs say about video loading."""
    print("\n📊 Analyzing training bottlenecks...")
    
    print("🔍 Key bottlenecks to investigate:")
    print("1. S3 download speed (should be <1s per video)")
    print("2. Video decoding (should be <2s for 40 frames)")
    print("3. Frame sampling/preprocessing (should be <0.5s)")
    print("4. Data collator batching (should be <1s)")
    print("5. Multiple process overhead (42 processes instead of 8)")
    
    print("\n🎯 Expected performance:")
    print("- Single video load: 2-5 seconds")
    print("- Batch of 8 videos: 10-20 seconds") 
    print("- Current actual: 10+ minutes (!)")
    
    print("\n⚠️  Likely culprits:")
    print("- S3 bandwidth throttling")
    print("- Process multiplication (42 vs 8 processes)")
    print("- Video decoding inefficiency")
    print("- Memory/disk I/O bottlenecks")

if __name__ == "__main__":
    print("🚀 Simplified Video Loading Performance Test")
    print("=" * 60)
    
    # Test S3 performance
    result = test_s3_download_speed()
    
    # Test connection settings
    test_s3_connection_settings()
    
    # Analysis
    analyze_training_logs()
    
    print("\n✅ Test completed!")
    
    if result:
        video_id, size, head_time, chunk_time = result
        print(f"\n📋 Summary for {video_id}:")
        print(f"  • File size: {size / (1024*1024):.1f} MB")
        print(f"  • Head object: {head_time:.3f}s")
        print(f"  • 1MB download: {chunk_time:.3f}s")
        
        if chunk_time > 1.0:
            print(f"  ⚠️  SLOW: 1MB took {chunk_time:.1f}s (expected <0.5s)")
            print(f"  🔧 Recommendations:")
            print(f"     - Check network connectivity to eu-west-1")
            print(f"     - Consider using eu-west-1 EC2 instances")
            print(f"     - Implement video caching/preprocessing")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Run this test on the training instance")
    print(f"  2. Compare local vs remote performance")
    print(f"  3. Fix process multiplication issue (42 -> 8)")
    print(f"  4. Consider video preprocessing pipeline") 