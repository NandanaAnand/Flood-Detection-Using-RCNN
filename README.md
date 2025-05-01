# Flood-Detection-Using-RCNN
run in the following sequence : 
custom_tf_data_generator.py
load_and_train_tf_model.py
train_model_tf_simple.py
twilio_sms_config.py
torch_dataset_loader.py
test_trained_model_and_webcam.py
water_level_monitor.py

1. Data Preparation
Scripts:
- `custom_tf_data_generator.py`
- `torch_dataset_loader.py`
- `load_and_train_tf_model.py`
- `train_model_tf_simple.py`

Purpose:
- Load raw images and segmentation masks.
- Preprocess (resize, normalize).
- Organize into batches using TensorFlow `tf.data.Dataset` or PyTorch `DataLoader`.

2. Model Definition & Training
Scripts:
- `load_and_train_tf_model.py`
- `train_model_tf_simple.py`

Purpose:
- Define a CNN model using Keras (TensorFlow).
- Train the model on preprocessed datasets.
- Save the trained model in `.keras` format (`water_contour_cnn.keras`).

3. Model Testing / Inference
Script:
- `test_trained_model_and_webcam.py`

Purpose:
- Load the saved trained model.
- Predict segmentation masks on test images or webcam frames.
- Display original vs predicted images side-by-side for verification.

4. Real-Time Flood Monitoring
Script:
- 'main.py' aka `water_level_monitor.py`

Purpose:
- Open live webcam feed.
- Detect water presence using color segmentation + pre-trained mask (optional).
- Check if water crosses predefined threshold lines (green = warning, red = critical).
- Trigger an alert if thresholds are crossed.

5. SMS Alert System
Script:
- `twilio_sms_config.py`

Purpose:
- Store Twilio API credentials.
- Enable other scripts to send SMS alerts when water levels are critical.

Summary Flowchart


[ Raw Images & Masks ]
          ↓
[ Data Loading & Preprocessing ]
          ↓
[ Model Definition & Training ]
          ↓
[ Save Trained Model (.keras) ]
          ↓
[ Test on Image or Webcam ]
          ↓
[ Real-time Monitoring via Webcam ]
          ↓
[ Water Level Detected? ]
          ↓
[ Threshold Crossed? ] ── Yes ──► [ Send SMS via Twilio ]
