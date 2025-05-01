"""
Description:Implements a PyTorch Dataset class to load and preprocess images and masks.
            Applies transformations using torchvision and prepares a DataLoader for training.
"""
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Paths to datasets
RAW_IMAGES_DIR = r"D:\Project files\riwa_v2\images"
MASKS_DIR = r"D:\Project files\riwa_v2\masks"

# Define dataset class
class WaterContourDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, transform=None):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(raw_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load raw image
        img_path = os.path.join(self.raw_dir, self.image_filenames[idx])
        img = cv2.imread(img_path)  # Read as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load corresponding mask
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Normalize mask (0 or 1)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=2)  # Make it (H, W, 1)

        # Apply transformations (resize, convert to tensor)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

# Define transformations (Resize to 256x256, Convert to Tensor, Normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Convert to tensor (C, H, W)
])

# Create dataset instance
dataset = WaterContourDataset(RAW_IMAGES_DIR, MASKS_DIR, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test data loading
for img, mask in dataloader:
    print(f"Image Shape: {img.shape}, Mask Shape: {mask.shape}")
    break  # Only show first batch
