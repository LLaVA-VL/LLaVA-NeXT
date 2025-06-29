#!/usr/bin/env python3
"""Test the complete LLaVA video loading pipeline locally to identify real bottlenecks."""

import time
import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

def test_complete_pipeline():
    """Test the complete video loading pipeline exactly as LLaVA does it."""
    print("🔍 Testing complete LLaVA video loading pipeline...")
    
    # Set up the environment like LLaVA training
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    
    try:
        # Import what we need 
        from llava.track_segment_loading import load_video_track_segment
        import torch
        
        # Test data - simulate a real training sample
        video_id = "00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4"  # From our S3 test
        track_id = 1
        timespan = (1.0, 10.0)  # 9 second segment
        num_frames = 40  # Same as training config
        
        print(f"📹 Video: {video_id}")
        print(f"🎯 Track: {track_id}, Timespan: {timespan}, Frames: {num_frames}")
        print("-" * 60)
        
        # Step 1: Video loading (like TrackSegmentDataset.__getitem__)
        print("Step 1: Video Loading...")
        step1_start = time.time()
        
        try:
            frame_batch = load_video_track_segment(video_id, track_id, timespan, num_frames)
            step1_time = time.time() - step1_start
            print(f"✅ Video loaded: {step1_time:.3f}s")
            print(f"   📐 Shape: {frame_batch.data.shape}")
            print(f"   🎬 PTS: [{frame_batch.pts_seconds[0]:.2f}, {frame_batch.pts_seconds[-1]:.2f}]s")
        except Exception as e:
            print(f"❌ Video loading failed: {e}")
            return
        
        # Step 2: Image preprocessing (like SigLIP processor)
        print("\nStep 2: Image Preprocessing...")
        step2_start = time.time()
        
        try:
            # Use the actual processor like in training
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            image_processor = processor.image_processor
            
            # This is the expensive step!
            processed_images = image_processor.preprocess(frame_batch.data, return_tensors="pt")["pixel_values"]
            step2_time = time.time() - step2_start
            print(f"✅ Images processed: {step2_time:.3f}s")
            print(f"   📐 Processed shape: {processed_images.shape}")
            print(f"   💾 Memory: {processed_images.numel() * processed_images.element_size() / 1e6:.1f} MB")
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            # Fallback to simple processing
            step2_start = time.time()
            processed_images = frame_batch.data.permute(0, 3, 1, 2).float() / 255.0
            step2_time = time.time() - step2_start
            print(f"⚠️  Used simple processing: {step2_time:.3f}s")
        
        # Step 3: Text processing (like conversation preprocessing)
        print("\nStep 3: Text Processing...")
        step3_start = time.time()
        
        # Simulate conversation preprocessing
        conversations = [
            {"from": "human", "value": "What is happening in this video?"},
            {"from": "gpt", "value": "This video shows a person walking in a room."}
        ]
        
        # Simulate tokenization (without actually loading tokenizer)
        sample_text = "".join([conv["value"] for conv in conversations])
        text_length = len(sample_text)
        step3_time = time.time() - step3_start
        print(f"✅ Text processed: {step3_time:.3f}s")
        print(f"   📝 Text length: {text_length} chars")
        
        # Step 4: Data collation simulation
        print("\nStep 4: Data Collation...")
        step4_start = time.time()
        
        # Simulate what DataCollatorForSupervisedDataset does
        batch_data = {
            "image": [(processed_images, frame_batch.data[0].size(), "video")],
            "conversations": conversations,
            "id": "test_sample"
        }
        step4_time = time.time() - step4_start
        print(f"✅ Data collated: {step4_time:.3f}s")
        
        # Total pipeline time
        total_time = step1_time + step2_time + step3_time + step4_time
        print(f"\n🏁 TOTAL PIPELINE TIME: {total_time:.3f} seconds")
        print(f"📊 Breakdown:")
        print(f"   • Video loading: {step1_time:.3f}s ({step1_time/total_time*100:.1f}%)")
        print(f"   • Image processing: {step2_time:.3f}s ({step2_time/total_time*100:.1f}%)")
        print(f"   • Text processing: {step3_time:.3f}s ({step3_time/total_time*100:.1f}%)")
        print(f"   • Data collation: {step4_time:.3f}s ({step4_time/total_time*100:.1f}%)")
        
        # Analysis
        print(f"\n🔍 Analysis:")
        if total_time > 10.0:
            print(f"❌ SLOW: {total_time:.1f}s is too slow for one sample!")
        elif total_time > 5.0:
            print(f"⚠️  SLOW: {total_time:.1f}s is borderline slow")
        else:
            print(f"✅ FAST: {total_time:.1f}s is acceptable")
        
        print(f"\n🎯 For batch of 8 videos:")
        batch_time = total_time * 8  # Sequential processing
        print(f"   • Sequential: {batch_time:.1f}s")
        print(f"   • Expected with 4 workers: {batch_time/4:.1f}s")
        
        return total_time, step1_time, step2_time
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Environment may not have all dependencies")
        return None, None, None

def test_multiple_videos():
    """Test loading multiple videos to simulate batch behavior."""
    print("\n" + "="*60)
    print("🎬 Testing Multiple Video Loading")
    print("="*60)
    
    # List of test videos
    test_videos = [
        ("00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4", 1, (1.0, 10.0)),
        ("00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4", 2, (2.0, 11.0)),  # Different track
        ("00003417-8588-5139-b465-ac976de5424a_exploitable_sequence_0_split_0.mp4", 1, (5.0, 14.0)),  # Different timespan
    ]
    
    try:
        from llava.track_segment_loading import load_video_track_segment
        
        total_start = time.time()
        load_times = []
        
        for i, (video_id, track_id, timespan) in enumerate(test_videos):
            print(f"\n📹 Video {i+1}/3: track {track_id}, timespan {timespan}")
            
            start = time.time()
            frame_batch = load_video_track_segment(video_id, track_id, timespan, 40)
            load_time = time.time() - start
            load_times.append(load_time)
            
            print(f"   ✅ Loaded: {load_time:.3f}s ({frame_batch.data.shape})")
        
        total_time = time.time() - total_start
        avg_time = sum(load_times) / len(load_times)
        
        print(f"\n📊 Multiple Video Results:")
        print(f"   • Total time: {total_time:.3f}s")
        print(f"   • Average per video: {avg_time:.3f}s") 
        print(f"   • Individual times: {[f'{t:.3f}s' for t in load_times]}")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ Multiple video test failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Complete LLaVA Video Loading Pipeline Test")
    print("=" * 60)
    
    # Test complete pipeline
    total, video_load, image_proc = test_complete_pipeline()
    
    # Test multiple videos
    avg_multi = test_multiple_videos()
    
    # Final analysis
    print("\n" + "="*60)
    print("🎯 FINAL ANALYSIS")
    print("="*60)
    
    if total:
        print(f"Single video complete pipeline: {total:.3f}s")
        if video_load: print(f"  - Video loading: {video_load:.3f}s")
        if image_proc: print(f"  - Image processing: {image_proc:.3f}s")
    
    if avg_multi:
        print(f"Average video loading time: {avg_multi:.3f}s")
    
    print(f"\n🔍 Expected vs Actual:")
    print(f"• Local single video: {total:.1f}s" if total else "• Local single video: FAILED")
    print(f"• Training batch (8 videos): 10+ minutes = 600+ seconds")
    print(f"• Expected batch time: {total*8:.1f}s" if total else "• Expected batch time: UNKNOWN")
    
    if total and total * 8 < 60:
        print(f"\n❗ CONCLUSION: Video loading should be fast!")
        print(f"The 10+ minute delay in training is NOT due to video loading itself.")
        print(f"Likely causes:")
        print(f"  - Process contention (43 processes)")
        print(f"  - Network bandwidth limits")
        print(f"  - Memory/disk I/O bottlenecks")
        print(f"  - DeepSpeed initialization overhead")
        print(f"  - Other distributed training issues")
    
    print("\n✅ Test completed!") 