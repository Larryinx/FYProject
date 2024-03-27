import torch
from PIL import Image
from clip import load
import clip
import faiss
import os
import numpy as np
import time
from decord import VideoReader, cpu

import classes

start_inference = time.perf_counter()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
model, preprocess = load('ViT-B/32', device=device)
print("Model loaded successfully")

# Your video path and text query
video_path = '../VideoDataAirport/amsterdam_airport_2.mp4'
text_query = "a men in orange coat"
class_list = [f" {c}" for c in classes.CLASSES]
semantic_search_phrase = class_list

# Function to load frames from the video
def load_images(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = [vr[i].asnumpy() for i in range(0, len(vr), 10)]
    return frames
print(f"Number of frames in the video: {len(load_images(video_path))}")

# Process and encode the frames
start_encoding = time.perf_counter()
image_vectors = []
for frame in load_images(video_path):
    image = Image.fromarray(frame)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(processed_image).squeeze(0)
    image_vectors.append(image_features.cpu().numpy())
end_encoding = time.perf_counter()
print(f"Encoding completed in {end_encoding - start_encoding:0.4f} seconds")
print(f"Number of frames processed: {len(image_vectors)}")

# Convert list of numpy arrays to a single numpy array
image_vectors = np.array(image_vectors)
print(f"Shape of image vectors: {image_vectors.shape}")


# Function to encode and normalize text queries
def encode_text_queries(queries):
    encoded_queries = []
    for query in queries:
        text_inputs = clip.tokenize(query).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs).cpu().numpy()
        faiss.normalize_L2(text_features)
        encoded_queries.append(text_features)
    return encoded_queries


# Encode all text queries (semantic search phrases + original query)
all_queries = semantic_search_phrase + [text_query]
start_encoding = time.perf_counter()
encoded_all_queries = encode_text_queries(all_queries)
end_encoding = time.perf_counter()
print(f"Encoding completed in {end_encoding - start_encoding:0.4f} seconds")

# Combine all encoded queries into a single numpy array
all_query_vectors = np.vstack(encoded_all_queries)
print(f"Shape of all query vectors: {all_query_vectors.shape}")

# Normalize and create a FAISS index for all text query vectors
index_queries = faiss.IndexFlatIP(all_query_vectors.shape[1])
faiss.normalize_L2(all_query_vectors)
index_queries.add(all_query_vectors)

# Shortlist frames based on the new criteria
shortlisted_frames = []
for frame_idx in range(len(image_vectors)):
    # Retrieve the feature vector of the candidate frame and normalize it
    frame_vector = np.array([image_vectors[frame_idx]])
    faiss.normalize_L2(frame_vector)

    # Perform the search for the closest text queries
    D, I = index_queries.search(frame_vector, k=20)  # Searching top 20 nearest queries

    # Find the index of the original query in the results
    original_query_index = len(semantic_search_phrase)  # Original query is added last
    if original_query_index in I[0]:
        shortlisted_frames.append(frame_idx)

# Display shortlisted frames
print(f"Number of frames in shortlist: {len(shortlisted_frames)}")
for frame_number in shortlisted_frames:
    print(f"Frame {frame_number * 10} is in the shortlist.")

end_inference = time.perf_counter()
print(f"Total inference time: {end_inference - start_inference:0.4f} seconds")