import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import classes

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import frames
import retrieved_frames
import outputs

frame_rate = 20 #@param {type:"integer"}
top_percentage = 30 #@param {type:"number"}
original_search_text = "a men in orange coat" #@param {type:"string"}
video_path = "../VideoDataAirport/amsterdam_airport_2.mp4" #@param {type:"string"}
folder_path = './img_folder_cat' #@param {type:"string"}
result_image_folder = './result_image_cat' #@param {type:"string"}

# Extract frames
print(f"Number of frames in the video: {frames.count_frames(video_path)}")
frames.extract_frames(video_path, frame_rate, folder_path)

# Load images
loaded_images, image_filenames, file_number_to_index = retrieved_frames.load_images_with_filenames(folder_path)
print("Total number of images retrieved: ", len(loaded_images))

# Prepare the semantic search phrase
class_list = [f" {c}" for c in classes.CLASSES]
semantic_search_phrase = [original_search_text]
additional_descriptors = ["blurry", "grainy", "low resolution", "foggy", "sepia"]
semantic_search_phrase += class_list + additional_descriptors


# Dictionary to store candidates and their scores
candidates = {}

# Process each image and check for candidates
for image, file_number in zip(loaded_images, file_number_to_index.values()):

    print(f"Processing image {file_number}")

    # Process the input
    inputs = processor(text=semantic_search_phrase, images=image, return_tensors="pt", padding=True)

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(logits)

    # Get top 20 labels
    top_20_probs, top_20_indices = torch.topk(probabilities, 20)
    top_20_labels = [semantic_search_phrase[index] for index in top_20_indices.squeeze().tolist()]

    # Check if the original search text is in the top 20 labels
    if original_search_text in top_20_labels:
        print(f"Original search text found in image {file_number}")
        score = probabilities[0, top_20_indices.squeeze().tolist().index(semantic_search_phrase.index(original_search_text))]
        candidates[file_number] = score.item()

# Order the candidates by score and select the top 30%
number_of_selected_images = int(len(candidates) * (top_percentage / 100))
top_percentage_scores = dict(sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:number_of_selected_images])

# Save the selected images using the save_image function
outputs.save_image(top_percentage_scores, folder_path, result_image_folder)
