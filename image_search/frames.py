import cv2
import os

# Function to count frames
def count_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0

    # Count the number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    return frame_count

# Modified extract_frames function
# Function to extract frames
def extract_frames(video_path, frame_rate, output_folder):
    # Read the video from specified path
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if the output folder is not empty
    if os.listdir(output_folder):
        print("Error: Image folder is not empty. If you want to overwrite, delete the folder 'img_folder' and run again.")
        return

    # Frame count
    frame_count = 0

    # Use a while loop to read frames
    while True:

        success, frame = video.read()

        # When video has ended or error occurs
        if not success:
            break

        # if frame_count < 3000 or frame_count > 4000:
        #     frame_count += 1
        #     continue

        # Capture one frame every 'frame_rate' frames
        if frame_count % frame_rate == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_name, frame)

        frame_count += 1

    video.release()
    print("Frames extracted successfully.")