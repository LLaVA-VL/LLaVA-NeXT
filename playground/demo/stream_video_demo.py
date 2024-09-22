import numpy as np
import cv2
import warnings
import select
import sys
import openai
import base64

warnings.filterwarnings("ignore")

# Global variables for storing video frames and their respective times
video_frames = []
frame_times = []
history_time = 0



client = openai.Client(api_key="EMPTY", base_url="xxx")

def encode_image(frames):
    base64_frames = []
    for frame in frames:
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert BGR to RGB
        _, buffer = cv2.imencode(".jpg", frame)
        buffer = base64.b64encode(buffer).decode("utf-8")
        base64_frames.append(buffer)
    return base64_frames

# Function to send frames to the server and get a response
def request_server(question, base64_frames):
    messages = [{"role": "user", "content": []}]
    for base64_frame in base64_frames:
        frame_format = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
            "modalities": "video",
        }
        messages[0]["content"].append(frame_format)

    prompt = {"type": "text", "text": question}
    messages[0]["content"].append(prompt)

    video_request = client.chat.completions.create(
        model="llava-onevision-72b-ov",
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )

    return video_request.choices[0].message.content


class Args:
    """
    Class to store configuration arguments.
    """
    def __init__(self, frame_limit=30, force_sample=False):
        self.frame_limit = frame_limit  # Max number of frames to retrieve
        self.force_sample = force_sample  # Whether to force uniform sampling


# Function to capture frames from the camera until the user presses Enter
def load_camera_frames_until_enter(args):
    global history_time  # To maintain across multiple captures

    cap = cv2.VideoCapture(0)  # 0 is the ID for the default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if unable to retrieve FPS
    frame_count = 0

    print("Video capturing started. Press 'Enter' in the console to stop capturing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame_count += 1
        cur_frame_time = frame_count / fps

        video_frames.append(frame)
        frame_times.append(cur_frame_time + history_time)

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Add cv2.waitKey to ensure the window remains visible
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if user pressed 'Enter' in the console
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input()  # Consume the "Enter" key press
            print("Video capture stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()  # Close the camera feed window

    history_time = frame_times[-1] if frame_times else history_time

    # Sample frames
    total_frames = len(video_frames)
    print(f"Total Frames Captured: {total_frames}")
    
    if total_frames > args.frame_limit:
        sample_indices = np.linspace(0, total_frames - 1, args.frame_limit, dtype=int)
        sampled_frames = [video_frames[i] for i in sample_indices]
        sampled_times = [frame_times[i] for i in sample_indices]
    else:
        sampled_frames = video_frames
        sampled_times = frame_times
    
    # import pdb; pdb.set_trace()
    frame_times_str = ",".join([f"{t:.2f}s" for t in sampled_times])
    return np.array(sampled_frames), frame_times_str, history_time


# Function to stream video, process it, and answer a user question
def stream_camera_and_ask_question(args):
    video_frames, frame_times, video_time = load_camera_frames_until_enter(args)

    if video_frames is None:
        print("Error capturing video frames.")
        return
    
    question = input("Press the query for current video: ").strip().lower()
    
    print("question: ", question)  
    image_base64 = encode_image(video_frames)
    # import pdb; pdb.set_trace()
    response = request_server(question, image_base64)

    print(f"Model's Answer: {response}")
    print(f"Video Duration: 0 to {video_time:.2f} seconds")
    print(f"Frame Times: {frame_times}")

    return response


# Main loop to keep the system running and waiting for user input
def main_loop():
    question = "Please describe this video."
    args = Args(frame_limit=64, force_sample=True)

    while True:
        answer = stream_camera_and_ask_question(args)
        if answer is None:
            print("Exiting the loop.")
            break
        
        user_input = input("Press 'Enter' to capture again, or 'q' to quit: ").strip().lower()
        if user_input == "q":
            print("Quitting the demo.")
            break

    # Close all OpenCV windows after the user quits
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
