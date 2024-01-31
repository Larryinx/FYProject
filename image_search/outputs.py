import shutil
import os

def save_image(top_percentage_scores, folder_path, result_image_folder):
    if not os.path.exists(result_image_folder):
        os.makedirs(result_image_folder, exist_ok=True)

    # Check if the output folder is not empty
    if os.listdir(result_image_folder):
        print("Error: Output folder is not empty. If you want to overwrite, delete the folder 'result_image' and run again.")
        return

    # Rank counter
    rank = 1

    # Copying the top scored images to result_image folder
    for file_number in top_percentage_scores.keys():
        # Construct the original file name from the file number
        original_file_name = f"frame_{file_number}.jpg"  # Assuming the file format is jpg
        original_file_path = os.path.join(folder_path, original_file_name)
        
        # Construct the new file name with rank
        new_file_name = f"top_{rank}_frame_{file_number}.jpg"
        destination_file_path = os.path.join(result_image_folder, new_file_name)

        # Copy the file
        shutil.copy2(original_file_path, destination_file_path)

        # Increment the rank
        rank += 1
    
    print("Top percentage images saved in the 'result_image' folder.")