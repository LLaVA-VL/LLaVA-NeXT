import cv2
import os
from decord import VideoReader, cpu
import numpy as np

id = 842
output_video_path = f"/Users/zhangyuanhan/Desktop/videomme_bad_shorts/{id}/{id}_origin.mp4"
# id = output_video_path.split("/")[-2]
output_saved_frames_path = output_video_path.replace(".mp4", "_sampled.mp4")

def load_video(video_path, max_frames_num=32, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time, video_time

def save_frames_as_video(frames, output_path, fps=1):
    # Get frame dimensions
    height, width, layers = frames[0].shape
    size = (width, height)
    # from BGR to RGB
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Initialize video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    # Write each frame
    for frame in frames:
        out.write(frame)
    
    out.release()

def save_frames_as_images(frames, id, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, frame in enumerate(frames):
        # from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_folder, f"{id}-frame_{i}.jpg"), frame)

spare_frames, frame_time, video_time = load_video(output_video_path, 512, force_sample=True)

# Save the sampled frames as a video
save_frames_as_video(spare_frames, output_saved_frames_path, fps=1)

save_frames_as_images(spare_frames,id, output_video_path.replace(".mp4", "_frames"))

print(spare_frames.shape)
print(f"Frame times: {frame_time}")
print(f"Video time: {video_time:.2f}s")
