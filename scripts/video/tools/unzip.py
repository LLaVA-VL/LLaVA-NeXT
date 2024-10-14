import cv2

# List of image ids
id_list = "/Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_220.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_221.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_222.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_223.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_224.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_225.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_226.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_227.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_228.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_229.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_230.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_231.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_232.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_233.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_234.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_235.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_236.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_237.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_238.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_239.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_240.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_241.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_242.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_243.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_244.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_245.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_246.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_247.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_248.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_249.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_250.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_251.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_252.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_253.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_254.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_255.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_256.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_257.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_258.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_259.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_260.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_261.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_262.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_263.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_264.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_265.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_266.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_267.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_268.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_269.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_270.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_271.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_272.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_273.jpg /Users/zhangyuanhan/Desktop/videomme_bad_shorts/842/842_origin_frames/842-frame_274.jpg".split(" ")
for i in range(len(id_list)):
    id_list[i] = id_list[i].split("/")[-1].rsplit(".", 1)[0]
print(id_list)

# Get the folder path from the first image ID
file_name = id_list[0].split("-")[0].split("_")[-1]
image_root = f'/Users/zhangyuanhan/Desktop/videomme_bad_shorts/{file_name}/{file_name}_origin_frames/'
image_path = f'{image_root}/{id_list[0]}' + '.jpg'

# Read the first image to get dimensions
img = cv2.imread(image_path)
height, width, _ = img.shape

# Output video path
output_video_path = f'/Users/zhangyuanhan/Desktop/videomme_bad_shorts/{file_name}/{file_name}.mp4'

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))

# Write each image to the video
for id in id_list:
    image_path = f'{image_root}/{id}.jpg'
    img = cv2.imread(image_path)
    if img is not None:
        video.write(img)
    else:
        print(f"Image {image_path} not found!")

# Release the video writer
video.release()

# Get the fps of the video
cap = cv2.VideoCapture(output_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")


