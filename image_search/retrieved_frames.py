import os
from PIL import Image

def load_images_with_filenames(folder_path):
    all_images = []
    all_filenames = []
    file_number_to_index = {}

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Assuming filenames are sortable numerically

    for index, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        all_images.append(image)
        all_filenames.append(filename)
        file_number = int(filename.split('_')[1].split('.')[0])  # Extract the number from the filename
        file_number_to_index[file_number] = index

    return all_images, all_filenames, file_number_to_index