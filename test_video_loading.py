#!/usr/bin/env python3
"""Test script to time video loading performance and identify bottlenecks."""

import time
import sys
import os
sys.path.append(os.path.dirname(__file__))

import boto3
from llava.track_segment_loading import load_video_track_segment
import torch

def get_sample_video_ids():
    """Get a few sample video IDs from S3 to test with."""
    print("🔍 Discovering sample video IDs from S3...")
    
    try:
        s3 = boto3.client('s3', region_name='eu-west-1')
        bucket = "scalable-training-dataset"
        prefix = "Simone_28_full/videos/"
        
        # List some videos from S3
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=20  # Get first 20 videos
        )
        
        video_ids = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.webm') or key.endswith('.mp4'):
                    video_id = key.replace(prefix, '')
                    video_ids.append(video_id)
                    if len(video_ids) >= 10:  # Limit to 10 for testing
                        break
        
        print(f"✅ Found {len(video_ids)} video files:")
        for i, vid in enumerate(video_ids[:5]):
            print(f"  {i+1}. {vid}")
        if len(video_ids) > 5:
            print(f"  ... and {len(video_ids) - 5} more")
        
        return video_ids
        
    except Exception as e:
        print(f"❌ Failed to list S3 videos: {e}")
        # Fallback to dummy IDs for structure testing
        print("📄 Using fallback test IDs...")
        return [
            "video1.webm", "video2.webm", "video3.webm"
        ]

def test_single_video_loading(video_ids):
    """Test loading a single video to identify performance bottlenecks."""
    print("\n🔬 Testing single video loading performance...")
    
    if not video_ids:
        print("❌ No video IDs available")
        return
        
    video_id = video_ids[0]
    track_id = 1
    timespan = (1.0, 10.0)  # 9 second segment
    num_frames = 40  # Same as training
    
    print(f"📹 Testing: {video_id}")
    print(f"🎯 Track ID: {track_id}")
    print(f"⏱️  Timespan: {timespan}")
    print(f"🎬 Frames: {num_frames}")
    print("-" * 50)
    
    # Time each step separately
    total_start = time.time()
    
    try:
        # Test S3 download time separately
        print("📥 Testing S3 download...")
        s3_start = time.time()
        
        s3 = boto3.client('s3', region_name='eu-west-1')
        bucket = "scalable-training-dataset"
        prefix = "Simone_28_full/videos/"
        
        # Test object exists and get size
        try:
            response = s3.head_object(Bucket=bucket, Key=prefix + video_id)
            file_size = response['ContentLength']
            print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"⚠️  Cannot access {video_id}: {e}")
            return
        
        # Test the full video loading pipeline
        load_start = time.time()
        frames = load_video_track_segment(video_id, track_id, timespan, num_frames)
        load_time = time.time() - load_start
        
        print(f"✅ Video loaded successfully!")
        print(f"⏱️  Total load time: {load_time:.3f} seconds")
        print(f"📐 Frame shape: {frames.data.shape}")
        print(f"🎬 PTS range: [{frames.pts_seconds[0]:.2f}, {frames.pts_seconds[-1]:.2f}]s")
        print(f"⚡ Loading rate: {len(frames.data) / load_time:.1f} frames/sec")
        
        # Test image preprocessing time
        print("\n🖼️  Testing image preprocessing...")
        preprocess_start = time.time()
        
        # Use a simpler processor to avoid downloading huge models
        # Just test the tensor operations
        frames_tensor = frames.data  # [N, H, W, C]
        
        # Simulate the preprocessing that SigLIP does
        # Convert from [N,H,W,C] to [N,C,H,W] and normalize
        frames_processed = frames_tensor.permute(0, 3, 1, 2).float() / 255.0
        
        preprocess_time = time.time() - preprocess_start
        
        print(f"⏱️  Preprocessing time: {preprocess_time:.3f} seconds")
        print(f"📐 Processed shape: {frames_processed.shape}")
        print(f"⚡ Preprocessing rate: {len(frames.data) / preprocess_time:.1f} frames/sec")
        
        total_time = time.time() - total_start
        print(f"\n🏁 Total pipeline time: {total_time:.3f} seconds")
        print(f"⚡ Overall rate: {len(frames.data) / total_time:.1f} frames/sec")
        
        # Analyze bottlenecks
        if load_time > 5.0:
            print(f"⚠️  WARNING: Video loading took {load_time:.1f}s - investigating...")
            print(f"   • S3 download might be slow")
            print(f"   • Video decoding might be inefficient")
            print(f"   • File size: {file_size / (1024*1024):.1f} MB")
            print(f"   • Download rate: {(file_size / (1024*1024)) / load_time:.1f} MB/s")
        
        return load_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_batch_loading(video_ids):
    """Test loading multiple videos to simulate batch behavior."""
    print("\n" + "="*60)
    print("🎬 Testing batch loading (8 videos)...")
    print("="*60)
    
    if len(video_ids) < 8:
        print(f"⚠️  Only {len(video_ids)} videos available, testing with those...")
        test_videos = video_ids
    else:
        test_videos = video_ids[:8]
    
    num_frames = 40
    successful_loads = 0
    total_start = time.time()
    load_times = []
    
    for i, video_id in enumerate(test_videos):
        print(f"\n📹 Loading video {i+1}/{len(test_videos)}: {video_id[:30]}...")
        try:
            start = time.time()
            frames = load_video_track_segment(video_id, 1, (1.0, 10.0), num_frames)
            load_time = time.time() - start
            successful_loads += 1
            load_times.append(load_time)
            print(f"  ✅ Loaded in {load_time:.3f}s ({len(frames.data)} frames)")
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}...")
    
    total_time = time.time() - total_start
    print(f"\n📊 Batch Results:")
    print(f"  • Successful loads: {successful_loads}/{len(test_videos)}")
    print(f"  • Total time: {total_time:.3f} seconds")
    print(f"  • Average per video: {total_time / len(test_videos):.3f} seconds")
    
    if load_times:
        print(f"  • Fastest load: {min(load_times):.3f}s")
        print(f"  • Slowest load: {max(load_times):.3f}s")
        print(f"  • Average load: {sum(load_times) / len(load_times):.3f}s")
    
    # Performance analysis
    expected_batch_time = 15  # seconds
    if total_time > expected_batch_time:
        print(f"\n⚠️  WARNING: Batch took {total_time:.1f}s - TOO SLOW!")
        print(f"     Expected: ~{expected_batch_time} seconds for {len(test_videos)} videos")
        print(f"     Actual: {total_time:.1f} seconds")
        print(f"     Per video average: {total_time / len(test_videos):.2f}s")
        
        if load_times:
            slow_videos = [t for t in load_times if t > 3.0]
            print(f"     Videos >3s: {len(slow_videos)}/{len(load_times)}")
    else:
        print(f"✅ Batch performance acceptable: {total_time:.1f}s")

if __name__ == "__main__":
    print("🔍 LLaVA Video Loading Performance Test")
    print("=" * 60)
    
    # Discover video IDs
    video_ids = get_sample_video_ids()
    
    if video_ids:
        # Test single video first
        single_load_time = test_single_video_loading(video_ids)
        
        # Test batch loading if single load works
        if single_load_time is not None:
            test_batch_loading(video_ids)
        else:
            print("❌ Skipping batch test due to single video failure")
    else:
        print("❌ No video IDs available for testing")
    
    print("\n✅ Performance test completed!") 