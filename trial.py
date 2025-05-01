import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from main import WaterContourCNN

# Load the trained model (ensure it's in eval mode)
model = WaterContourCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Load and preprocess an image
image_path = "D:/Nannu/Raw_Images/frame_0.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = cv2.resize(image, (256, 256))  # Resize to model's expected size
image = image.astype(np.float32) / 255.0  # Normalize

# Convert to tensor (PyTorch format: [batch, channels, height, width])
image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

# Run the model
with torch.no_grad():
    output_mask = model(image_tensor)

# Convert output to binary mask
output_mask = output_mask.squeeze().numpy()  # Remove batch dim, convert to numpy
binary_mask = (output_mask > 0.5).astype(np.uint8) * 255  # Thresholding

# Show input vs output
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Predicted Water Contour")
plt.axis("off")

plt.show()
