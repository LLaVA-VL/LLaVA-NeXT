#!/usr/bin/env python3
"""Direct video loading test - bypass LLaVA imports to test core video loading."""

import time
import boto3
import torch
import io

def test_video_loading_steps():
    """Test each step of video loading separately."""
    print("🔍 Testing video loading steps separately...")
    
    # Test config
    bucket = "scalable-training-dataset"
    prefix = "Simone_28_full/videos/"
    video_id = "00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4"
    
    print(f"📹 Video: {video_id}")
    print("-" * 60)
    
    try:
        # Step 1: S3 client setup
        print("Step 1: S3 Client Setup...")
        s3_setup_start = time.time()
        s3 = boto3.client('s3', region_name='eu-west-1')
        s3_setup_time = time.time() - s3_setup_start
        print(f"✅ S3 client: {s3_setup_time:.3f}s")
        
        # Step 2: S3 metadata check
        print("\nStep 2: S3 Metadata Check...")
        meta_start = time.time()
        response = s3.head_object(Bucket=bucket, Key=prefix + video_id)
        file_size = response['ContentLength']
        meta_time = time.time() - meta_start
        print(f"✅ Metadata: {meta_time:.3f}s")
        print(f"   📊 File size: {file_size / (1024*1024):.1f} MB")
        
        # Step 3: S3 download
        print("\nStep 3: S3 Full Download...")
        download_start = time.time()
        response = s3.get_object(Bucket=bucket, Key=prefix + video_id)
        video_data = response['Body'].read()
        download_time = time.time() - download_start
        download_speed = (file_size / (1024*1024)) / download_time
        print(f"✅ Download: {download_time:.3f}s ({download_speed:.1f} MB/s)")
        
        # Step 4: Video decoding (if torchcodec available)
        print("\nStep 4: Video Decoding...")
        decode_start = time.time()
        
        try:
            import torchcodec
            from torchcodec.decoders import VideoDecoder
            
            # Create seekable file object
            video_file = io.BytesIO(video_data)
            decoder = VideoDecoder(video_file, device='cpu')
            
            # Decode timespan
            frames = decoder.get_frames_played_in_range(1.0, 10.0)  # 9 seconds
            decode_time = time.time() - decode_start
            
            print(f"✅ Decoding: {decode_time:.3f}s")
            print(f"   📐 Frames shape: {frames.data.shape}")
            print(f"   🎬 Frame count: {len(frames.data)}")
            
            # Step 5: Frame sampling 
            print("\nStep 5: Frame Sampling...")
            sample_start = time.time()
            
            num_frames_wanted = 40
            total_frames = len(frames.data)
            if total_frames > num_frames_wanted:
                interval = (total_frames - 1) / (num_frames_wanted - 1)
                indices = [round(i * interval) for i in range(num_frames_wanted)]
                sampled_frames = frames.data[indices]
            else:
                sampled_frames = frames.data
            
            sample_time = time.time() - sample_start
            print(f"✅ Sampling: {sample_time:.3f}s")
            print(f"   📐 Final shape: {sampled_frames.shape}")
            
            # Step 6: Basic image preprocessing 
            print("\nStep 6: Basic Image Preprocessing...")
            preprocess_start = time.time()
            
            # Convert from [N,H,W,C] to [N,C,H,W] and normalize
            processed = sampled_frames.permute(0, 3, 1, 2).float() / 255.0
            preprocess_time = time.time() - preprocess_start
            print(f"✅ Preprocessing: {preprocess_time:.3f}s")
            print(f"   📐 Processed shape: {processed.shape}")
            
            # Total time
            total_time = s3_setup_time + meta_time + download_time + decode_time + sample_time + preprocess_time
            print(f"\n🏁 TOTAL VIDEO PIPELINE: {total_time:.3f} seconds")
            print(f"📊 Breakdown:")
            print(f"   • S3 setup: {s3_setup_time:.3f}s ({s3_setup_time/total_time*100:.1f}%)")
            print(f"   • Metadata: {meta_time:.3f}s ({meta_time/total_time*100:.1f}%)")
            print(f"   • Download: {download_time:.3f}s ({download_time/total_time*100:.1f}%)")
            print(f"   • Decoding: {decode_time:.3f}s ({decode_time/total_time*100:.1f}%)")
            print(f"   • Sampling: {sample_time:.3f}s ({sample_time/total_time*100:.1f}%)")
            print(f"   • Preprocessing: {preprocess_time:.3f}s ({preprocess_time/total_time*100:.1f}%)")
            
            return total_time, download_time, decode_time
            
        except ImportError:
            print("❌ torchcodec not available")
            return None, download_time, None
        except Exception as e:
            print(f"❌ Decoding failed: {e}")
            return None, download_time, None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_concurrent_loading():
    """Test loading multiple videos concurrently to simulate training."""
    print("\n" + "="*60)
    print("🎬 Testing Concurrent Video Loading")
    print("="*60)
    
    # Use threading to simulate concurrent access
    import threading
    import queue
    
    bucket = "scalable-training-dataset"
    prefix = "Simone_28_full/videos/"
    video_id = "00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4"
    
    def load_video_worker(worker_id, results_queue):
        """Worker function to load a video."""
        try:
            start_time = time.time()
            
            # Each worker gets its own S3 client
            s3 = boto3.client('s3', region_name='eu-west-1')
            
            # Download video
            response = s3.get_object(Bucket=bucket, Key=prefix + video_id)
            video_data = response['Body'].read()
            
            load_time = time.time() - start_time
            results_queue.put((worker_id, load_time, len(video_data)))
            
        except Exception as e:
            results_queue.put((worker_id, None, str(e)))
    
    # Test with different numbers of concurrent workers
    for num_workers in [1, 4, 8]:
        print(f"\n🧪 Testing with {num_workers} concurrent workers...")
        
        results_queue = queue.Queue()
        threads = []
        
        # Start timing
        test_start = time.time()
        
        # Start workers
        for i in range(num_workers):
            thread = threading.Thread(target=load_video_worker, args=(i, results_queue))
            thread.start()
            threads.append(thread)
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        test_total_time = time.time() - test_start
        
        # Collect results
        load_times = []
        errors = []
        
        while not results_queue.empty():
            worker_id, load_time, data_or_error = results_queue.get()
            if load_time is not None:
                load_times.append(load_time)
            else:
                errors.append(f"Worker {worker_id}: {data_or_error}")
        
        # Report results
        if load_times:
            avg_time = sum(load_times) / len(load_times)
            max_time = max(load_times)
            min_time = min(load_times)
            
            print(f"   ✅ {len(load_times)}/{num_workers} workers succeeded")
            print(f"   ⏱️  Total time: {test_total_time:.3f}s")
            print(f"   📊 Per-worker times: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
            print(f"   🚀 Throughput: {len(load_times) / test_total_time:.2f} videos/sec")
            
            if num_workers > 1 and max_time > avg_time * 2:
                print(f"   ⚠️  High variance - possible contention")
                
        if errors:
            print(f"   ❌ {len(errors)} workers failed:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"      {error}")

def analyze_training_bottleneck():
    """Analyze what could cause 10+ minute delays in training."""
    print("\n" + "="*60)
    print("🔍 TRAINING BOTTLENECK ANALYSIS")
    print("="*60)
    
    print("🎯 Expected performance (local test):")
    print("  • Single video load: ~1-2 seconds")
    print("  • Batch of 8 videos: ~8-16 seconds")
    print("  • With 4 dataloader workers: ~2-4 seconds")
    
    print("\n❗ Actual training performance:")
    print("  • First batch: 10+ minutes = 600+ seconds")
    print("  • 43 processes instead of 8")
    
    print("\n🔍 Possible causes:")
    print("  1. 🔄 Process multiplication:")
    print("     - 8 GPU processes × 4 dataloader workers = 32+ processes")
    print("     - Each process downloading the same videos simultaneously")
    print("     - S3 bandwidth divided among all processes")
    
    print("  2. 🌐 Network bottlenecks:")
    print("     - Training instance in us-east-1, S3 bucket in eu-west-1")
    print("     - Cross-region bandwidth limits")
    print("     - Concurrent requests overwhelming S3 rate limits")
    
    print("  3. 💾 Memory/disk bottlenecks:")
    print("     - 43 processes loading 3-4MB videos simultaneously")
    print("     - Memory pressure causing swapping")
    print("     - Disk I/O saturation")
    
    print("  4. 🔒 Distributed training coordination:")
    print("     - All ranks waiting for rank 0 to finish first batch")
    print("     - DeepSpeed parameter synchronization overhead")
    print("     - NCCL communication delays")
    
    print("\n🎯 Next steps to verify:")
    print("  1. Run this test on the training instance")
    print("  2. Compare single vs multi-process performance")
    print("  3. Check network bandwidth to eu-west-1") 
    print("  4. Monitor memory/disk usage during loading")
    print("  5. Test with dataloader_num_workers=0")

if __name__ == "__main__":
    print("🚀 Direct Video Loading Performance Test")
    print("=" * 60)
    
    # Test individual video loading steps
    total_time, download_time, decode_time = test_video_loading_steps()
    
    # Test concurrent loading
    test_concurrent_loading()
    
    # Analyze training bottleneck
    analyze_training_bottleneck()
    
    # Final summary
    if total_time:
        print(f"\n📋 LOCAL PERFORMANCE SUMMARY:")
        print(f"  • Single video pipeline: {total_time:.3f}s")
        print(f"  • Expected batch of 8: {total_time * 8:.1f}s")
        print(f"  • Training actual: 600+ seconds")
        print(f"  • Performance gap: {600 / (total_time * 8):.0f}x slower!")
    
    print("\n✅ Test completed!") 