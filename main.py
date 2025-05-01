"""
 Description: Uses computer vision to detect water levels in a live webcam feed,
            applies a mask, detects waterlines, and triggers SMS alerts based on threshold crossings.
            Uses Twilio for notifications.
You can also name this as "water_level_monitor.py"
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from twilio.rest import Client
import twilio_sms_config

# Fixed threshold line coordinates
green_line_start = (530, 475)
green_line_end = (780, 475)
#green_line_start = (270, 360)
#green_line_end = (390, 360)

red_line_start = (530, 300)
red_line_end = (780, 300)
#red_line_start = (270, 200)
#red_line_end = (390, 200)

# Open webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Load and process mask
mask = cv2.imread(r"/mask.png")
if mask is None:
    print("Error: Mask image not found or unreadable.")
    cap.release()
    exit()

# Read the first frame to determine frame size
success, img = cap.read()
if not success:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

# Resize mask to match frame size
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# Convert mask to 3-channel if grayscale
if len(mask.shape) == 2:
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

plt.ion()  # Turn on interactive mode for real-time updates

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to read frame.")
        break

    # Apply mask
    imgRegion = cv2.bitwise_and(img, mask)

    # Convert to HSV color space
    hsv = cv2.cvtColor(imgRegion, cv2.COLOR_BGR2HSV)

    # Define color range for dark blue water
    lower_blue = np.array([100, 100, 50])   # Adjusted for darker blue
    upper_blue = np.array([130, 255, 255])

    # Create a mask for dark blue water
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of dark blue water
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the highest detected waterline
    waterline_y = None
    if contours:
        all_points = np.vstack(contours)  # Stack all contour points
        min_y = np.min(all_points[:, 0, 1])  # Get the topmost Y coordinate
        waterline_y = min_y

    # Draw threshold lines
    cv2.line(imgRegion, green_line_start, green_line_end, (0, 255, 0), 5)  # Green Line
    cv2.line(imgRegion, red_line_start, red_line_end, (0, 0, 255), 5)  # Red Line

    # Draw detected waterline
    if waterline_y is not None:
        cv2.line(imgRegion, (200, waterline_y), (1100, waterline_y), (255, 255, 0), 2)  # Cyan Waterline

        # Check threshold crossings
        if waterline_y <= red_line_start[1]:  # If waterline is above red line
            print("⚠️ Flooding imminent!")
            client = Client(keys_sms.account_sid, keys_sms.auth_token)
            message = client.messages.create(

                body="Threshold Crossed!",
                from_=keys_sms.twilio_number,
                to=keys_sms.target_number
            )
        elif waterline_y <= green_line_start[1]:  # If waterline is above green line
            print("⚠️ Moderate levels of water.")

        elif waterline_y >= green_line_start[1]:  # If waterline is below green line
            print("⚠️ Subpar levels of water.")

    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(imgRegion, cv2.COLOR_BGR2RGB)

    # Display frame using Matplotlib
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Waterline at Y: {waterline_y}" if waterline_y else "No Waterline Detected")
    plt.show(block=False)
    plt.pause(0.01)  # Small delay to update the plot
    plt.clf()  # Clear previous frame

cap.release()
plt.close()
