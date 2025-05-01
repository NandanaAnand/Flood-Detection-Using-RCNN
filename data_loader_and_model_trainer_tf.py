"""
Description:This script loads images and masks from local directories,
            converts them into a TensorFlow dataset,
            defines a CNN model using Keras,
            compiles and trains it,
            and then saves the trained model in Keras format.
"""

import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


# Paths
RAW_IMAGES_PATH = r"D:\Project files\riwa_v2\images"
MASKS_PATH = r"D:\Project files\riwa_v2\masks"

IMG_SIZE = (256, 256)  # Resize all images to 256x256

# Load images
def load_image(image_path):
    image = cv2.imread(image_path)  # Read image
    image = cv2.resize(image, IMG_SIZE)  # Resize
    image = image / 255.0  # Normalize (0 to 1)
    return image

# Load masks
def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask (grayscale)
    mask = cv2.resize(mask, IMG_SIZE)  # Resize
    mask = mask / 255.0  # Normalize (0 to 1)
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
    return mask

# Get dataset file names
image_filenames = sorted(os.listdir(RAW_IMAGES_PATH))
mask_filenames = sorted(os.listdir(MASKS_PATH))

# Load dataset into NumPy arrays
images = np.array([load_image(os.path.join(RAW_IMAGES_PATH, fname)) for fname in image_filenames])
masks = np.array([load_mask(os.path.join(MASKS_PATH, fname)) for fname in mask_filenames])

print(f"✅ Dataset Loaded: {len(images)} images and {len(masks)} masks")



##################################################3
BATCH_SIZE = 8  # Number of images per training batch

# Convert to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((images, masks))
dataset = dataset.shuffle(len(images)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Dataset prepared for training!")


# Define the model
def build_model():
    inputs = keras.Input(shape=(256, 256, 3))  # Adjust size if needed

    # Encoder (Feature Extraction)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Decoder (Segmentation Output)
    x = layers.Conv2DTranspose(64, (2, 2), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (2, 2), strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2DTranspose(1, (2, 2), strides=2, activation="sigmoid", padding="same")(x)

    return keras.Model(inputs, outputs)

# Build the model
model = build_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("✅ Model defined and compiled successfully!")

# Now you're ready to train! 🎯

# Assuming images and masks are NumPy arrays already loaded
# If you haven't loaded them yet, replace this with actual image/mask loading code.
# Example: images = np.array([...]), masks = np.array([...])

dataset = tf.data.Dataset.from_tensor_slices((images, masks))

# Shuffle, batch, and prefetch for optimized training
dataset = dataset.shuffle(100).batch(8).prefetch(tf.data.experimental.AUTOTUNE)

print("✅ Dataset successfully created!")

# Now train your model
history = model.fit(dataset, epochs=3, verbose=1)

model.save("water_contour_cnn.keras")
print("✅ Model saved successfully in Keras format!")


