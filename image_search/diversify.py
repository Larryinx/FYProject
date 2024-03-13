import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compare_image(image1, image2):
    cos = torch.nn.CosineSimilarity(dim=0)

    image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
    image1_features = model.encode_image( image1_preprocess)
    # ouput image 1 features
    print(image1_features)
    print(image1_features.shape)

    image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
    image2_features = model.encode_image( image2_preprocess)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(image1_features, image2_features).item()
    similarity = (similarity+1)/2
    return similarity

image1 = "./result_image/top_1_frame_4920.jpg"
image2= "./result_image/top_10_frame_460.jpg"


sim = compare_image(image1, image2)

# print("Image similarity", sim)