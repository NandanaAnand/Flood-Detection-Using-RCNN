"""
 Description:Defines a custom Keras Sequence class (WaterContourDataset)
            for batch loading and preprocessing of image and mask data,
            enabling memory-efficient training.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

# Folder paths
raw_images_folder = r"D:\Project files\riwa_v2\images"
masked_images_folder = r"D:\Project files\riwa_v2\masks"

# Image dimensions
IMG_SIZE = (256, 256)


class WaterContourDataset(Sequence):
    def __init__(self, image_folder, mask_folder, batch_size=8):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, index):
        batch_images = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]

        images, masks = [], []

        for img_name, mask_name in zip(batch_images, batch_masks):
            img_path = os.path.join(self.image_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)

            # Load and preprocess images
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize to [0,1]

            # Load and preprocess masks
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, IMG_SIZE) / 255.0  # Normalize
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)


# Create dataset loader
train_dataset = WaterContourDataset(raw_images_folder, masked_images_folder, batch_size=8)

# Verify dataset
sample_images, sample_masks = train_dataset[0]
print(f"Image shape: {sample_images.shape}, Mask shape: {sample_masks.shape}")
