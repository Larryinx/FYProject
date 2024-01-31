import cv2
import os
from transformers import CLIPProcessor, CLIPModel

def extract_frames(video_path, frame_rate, output_folder="img_folder"):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_rate == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_name, frame)
        frame_count += 1

    video.release()
    print("Frames extracted successfully.")