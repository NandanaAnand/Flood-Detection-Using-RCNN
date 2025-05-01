""" Description:Loads a pre-trained TensorFlow/Keras model
                and performs segmentation on a static image or a live webcam stream.
                Includes preprocessing, prediction, and visualization
                using both Matplotlib and OpenCV.
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model(r"D:\Project files\water_contour_cnn.keras")  # Update with your model path

# Function to preprocess input images
def preprocess_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, img_size)  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict mask for a single image
def predict_mask(image_path):
    img = preprocess_image(image_path)
    mask_pred = model.predict(img)[0]  # Get prediction
    mask_pred = (mask_pred > 0.5).astype(np.uint8)  # Convert to binary mask

    print("Debug: Mask Prediction Shape:", mask_pred.shape)  # Debugging print
    print("Debug: Mask Prediction Max Value:", mask_pred.max())

    # Show results
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_pred, cmap="gray")
    plt.title("Predicted Water Contour")
    plt.axis("off")

    plt.show()

# Function to process a live webcam feed
def run_webcam():
    cap = cv2.VideoCapture(1)  # Change to 1 if your main webcam is not detected

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Error: Cannot access webcam")
            break

        # Preprocess frame
        img = cv2.resize(frame, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict water contour
        mask_pred = model.predict(img)[0]
        mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255  # Convert to binary mask

        print("Debug: Webcam Frame Shape:", frame.shape)  # Debugging print
        print("Debug: Mask Shape:", mask_pred.shape)

        # Resize back to original webcam size
        mask_pred = cv2.resize(mask_pred, (frame.shape[1], frame.shape[0]))

        # Convert mask to color and overlay
        mask_colored = cv2.merge([mask_pred, mask_pred, np.zeros_like(mask_pred)])
        overlay = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)  # Blend

        # Show results using Matplotlib (alternative if OpenCV fails)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Webcam Water Contour Detection")
        plt.axis("off")
        plt.pause(0.001)  # Needed to update the Matplotlib figure

        # Show with OpenCV as well (comment out if it crashes)
        cv2.imshow("Water Contour Detection", overlay)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Uncomment one of the following lines to run the desired function ===

# Test on a single image
# predict_mask("D:/Nannu_trials/test_image.jpg")

# Run real-time water contour detection on webcam
run_webcam()
