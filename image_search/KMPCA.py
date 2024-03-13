import os
from PIL import Image
import torch
import clip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to process and encode images and parse their ranks
def encode_images(image_folder):
    image_features = []
    image_info = []  # To store rank and frame number
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.jpg'):
            # Parse rank and frame number from file name
            parts = img_file.split('_')
            rank = int(parts[1])
            frame_number = int(parts[3].split('.')[0])

            image_path = os.path.join(image_folder, img_file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image)
                image_features.append(image_feature.cpu().numpy())
                image_info.append((rank, frame_number, img_file))
    return image_features, image_info

# Load and encode images
image_folder = 'result_image'  # Change to your folder path
encoded_images, image_info = encode_images(image_folder)

# Flatten the feature vectors
flat_features = [features.flatten() for features in encoded_images]

# PCA for dimensionality reduction
pca = PCA(n_components=0.9)
reduced_features = pca.fit_transform(flat_features)

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(reduced_features)

# Find top-ranked image in each cluster
top_images = {}
for i, cluster in enumerate(clusters):
    rank, frame_number, img_file = image_info[i]
    if cluster not in top_images or top_images[cluster][0] > rank:
        top_images[cluster] = (rank, frame_number, img_file)

# Save top-ranked images in each cluster
output_folder = 'result_cla'
os.makedirs(output_folder, exist_ok=True)

for cluster, (rank, frame_number, img_file) in top_images.items():
    image = Image.open(os.path.join(image_folder, img_file))
    new_file_name = f"top_{rank}_frame_{frame_number}_{cluster}.jpg"
    image.save(os.path.join(output_folder, new_file_name))

print("Top-ranked images saved in each cluster.")

# Visualize after PCA reduction
plt.figure(figsize=(8, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.title("Data after PCA Reduction")
plt.show()

# Visualize data after classification
plt.figure(figsize=(8, 8))
for i in range(10):  
    plt.scatter(reduced_features[clusters == i, 0], reduced_features[clusters == i, 1], label=f'Cluster {i}')
plt.title("Data after K-means Clustering")
plt.legend()
plt.show()
