import os
from decord import VideoReader
from decord import cpu, gpu

def load_images(video_path):
    # Load the video
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(range(0, len(vr), 10))  # extract every 10th frame

    return frames

if __name__ == '__main__':
    video_path = '../VideoDataAirport/amsterdam_airport_2.mp4'
    frames = load_images(video_path)
    print(f"Number of frames extracted: {len(frames)}")
    print(f"Shape of each frame: {frames[0].shape}")