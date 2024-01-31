import os
import time
from PIL import Image
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel

class ProxyAnalyzer:
    def __init__(self, semantic_search_phrase, video_path, output_folder="img_folder", frame_rate=20, top_percentage=10):
        self.semantic_search_phrase = semantic_search_phrase
        self.frame_rate = frame_rate
        self.video_path = video_path
        self.output_folder = output_folder
        self.top_percentage = top_percentage
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def count_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    @staticmethod
    def load_images_with_filenames(folder_path):
        all_images, all_filenames = [], []
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            all_images.append(image)
            all_filenames.append(filename)
        return all_images, all_filenames

    def analyze_frames(self):
        start_time = time.perf_counter()
        loaded_images, image_filenames = self.load_images_with_filenames(self.output_folder)
        with torch.no_grad():
            inputs = self.processor(text=[self.semantic_search_phrase], images=loaded_images, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        values, indices = logits_per_image.squeeze().topk(int(len(logits_per_image) * self.top_percentage / 100))
        top_images, top_scores, top_filenames = [], [], []
        for score, index in zip(values, indices):
            idx = int(index.numpy())
            top_images.append(loaded_images[idx])
            top_scores.append(round(score.numpy().tolist(), 50))
            top_filenames.append(image_filenames[idx])
        self.top_percentage_scores = self.get_top_percentage_scores(logits_per_image)
        end_time = time.perf_counter()
        print(f"Analysis completed in {end_time - start_time} seconds.")
        return top_images, top_scores, top_filenames

    def get_top_percentage_scores(self, logits_per_image):
        file_number_to_score = {}
        image_files = sorted(os.listdir(self.output_folder), key=lambda f: int(f.split('_')[1].split('.')[0]))

        for index, filename in enumerate(image_files):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_number = int(filename.split('_')[1].split('.')[0])
                if index < logits_per_image.size(0):
                    file_number_to_score[file_number] = logits_per_image[index].item()
                else:
                    file_number_to_score[file_number] = None

        # Sort the dictionary by score in descending order and convert to a list of tuples
        sorted_scores = sorted(file_number_to_score.items(), key=lambda item: item[1], reverse=True)

        # Calculate the number of items to include for the specified top percentage
        top_count = int(len(sorted_scores) * (self.top_percentage / 100))

        # Create a new dictionary with top percentage scores
        top_percentage_scores = dict(sorted_scores[:top_count])

        return top_percentage_scores