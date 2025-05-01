"""
Description:Another TensorFlow training script that loads and normalizes data,
            defines a CNN model, trains it for 5 epochs, and saves the trained model.
            Similar to load_and_train_tf_model.py, but simpler and more focused.
"""

import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers

# ================================
# STEP 1: Load Images and Masks
# ================================

# Define dataset paths
raw_images_folder = r"D:\Project files\riwa_v2\images"
masked_images_folder = r"D:\Project files\riwa_v2\masks"

# Get sorted list of image files
image_files = sorted(os.listdir(raw_images_folder))
mask_files = sorted(os.listdir(masked_images_folder))

# Load images and masks into arrays
images, masks = [], []

for img_file, mask_file in zip(image_files, mask_files):
    # Load raw image
    img_path = os.path.join(raw_images_folder, img_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))  # Resize to match model input
    img = img / 255.0  # Normalize to [0,1]

    # Load mask image (grayscale)
    mask_path = os.path.join(masked_images_folder, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))  # Resize
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
    mask = mask / 255.0  # Normalize

    images.append(img)
    masks.append(mask)

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, masks))
dataset = dataset.batch(8).shuffle(100)
print(f"✅ Loaded {len(images)} images and masks successfully!")


# ================================
# STEP 2: Define the CNN Model
# ================================

def build_model():
    inputs = keras.Input(shape=(256, 256, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Decoder
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding="same", activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = build_model()
model.summary()
print("✅ Model defined and compiled successfully!")

# ================================
# STEP 3: Train the Model
# ================================

history = model.fit(dataset, epochs=5, verbose=1)

# Save the model in Keras format
model.save("D:\Project files\water_contour_cnn.keras")
print("✅ Model saved successfully in Keras format!")
