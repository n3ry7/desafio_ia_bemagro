import os
import random
import shutil

# Set the paths to the source directories
image_dir = "dataset/images"
mask_dir = "dataset/binary_masks"

# Set the paths to the destination directories
train_image_dir = "dataset/train_images"
train_mask_dir = "dataset/train_masks"
val_image_dir = "dataset/val_images"
val_mask_dir = "dataset/val_masks"

# Get a list of all image files in the source directory
image_files = os.listdir(image_dir)

# Select 10% of the images at random
num_images = len(image_files)
num_val_images = int(num_images * 0.1)
val_indices = random.sample(range(num_images), num_val_images)

# Create the destination directories
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# Copy the images and masks to the destination directories
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, os.path.splitext(image_file)[0] + "_plant_mask.png")
    if i in val_indices:
        shutil.copy(image_path, os.path.join(val_image_dir, image_file))
        shutil.copy(
            mask_path,
            os.path.join(val_mask_dir, os.path.splitext(image_file)[0] + "_plant_mask.png"),
        )
    else:
        shutil.copy(image_path, os.path.join(train_image_dir, image_file))
        shutil.copy(
            mask_path,
            os.path.join(train_mask_dir, os.path.splitext(image_file)[0] + "_plant_mask.png"),
        )

print("Data split complete.")
