import torch
from transformers import CLIPProcessor, CLIPModel
import os

import frames
import retrieved_frames
import outputs

frame_rate = 20 #@param {type:"integer"}
top_percentage = 30 #@param {type:"number"}
semantic_search_phrase = "a men in orange coat" #@param {type:"string"}
video_path = "../VideoDataAirport/amsterdam_airport_2.mp4" #@param {type:"string"}
folder_path = './img_folder' #@param {type:"string"}
result_image_folder = './result_image' #@param {type:"string"}

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def run_model():
    with torch.no_grad():
        inputs = processor(text=[semantic_search_phrase],
                        images=loaded_images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
    return outputs

def map_scores_to_file_numbers(folder_path, logits_per_image):
    file_number_to_score = {}
    image_files = sorted(os.listdir(folder_path), key=lambda f: int(f.split('_')[1].split('.')[0]))

    for index, filename in enumerate(image_files):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_number = int(filename.split('_')[1].split('.')[0])
            if index < logits_per_image.size(0):
                file_number_to_score[file_number] = logits_per_image[index].item()
            else:
                file_number_to_score[file_number] = None

    return file_number_to_score

def get_top_percentage_scores(file_number_to_score, top_percentage):
    # Ensure the top_percentage is between 0 and 100
    if top_percentage < 0 or top_percentage > 100:
        raise ValueError("top_percentage must be between 0 and 100")

    # Sort the dictionary by score in descending order and convert to a list of tuples
    sorted_scores = sorted(file_number_to_score.items(), key=lambda item: item[1], reverse=True)

    # Calculate the number of items to include for the specified top percentage
    top_count = int(len(sorted_scores) * (top_percentage / 100))

    # Create a new dictionary with top percentage scores
    top_percentage_scores = dict(sorted_scores[:top_count])

    return top_percentage_scores

# Extract frames
print(f"Number of frames in the video: {frames.count_frames(video_path)}")
frames.extract_frames(video_path, frame_rate, folder_path)
# Load images
loaded_images, image_filenames, file_number_to_index = retrieved_frames.load_images_with_filenames(folder_path)
print("Total no of images retrieved: ", len(loaded_images))

logits_per_image = run_model().logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print("logits per image shape:", logits_per_image.shape)
print("Label probs shape:", probs.shape)

# Get the top percentage scores
file_number_to_score = map_scores_to_file_numbers(folder_path, logits_per_image)
top_percentage_scores = get_top_percentage_scores(file_number_to_score, top_percentage)

# Printing the top percentage scores
for file_number, score in top_percentage_scores.items():
    print(f"File Number: {file_number}, Score: {score}")

# Output the top percentage images
outputs.save_image(top_percentage_scores, folder_path, result_image_folder)