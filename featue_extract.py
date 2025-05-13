import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image

def extract_features_tf_grayscale(frames_folder, skip_frames=1, resize_dim=(224, 224)):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.tif')])
    features = []

    for i, file in enumerate(frame_files):
        if i % skip_frames != 0:
            continue

        img_path = os.path.join(frames_folder, file)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, resize_dim)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img_array = image.img_to_array(rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)
        features.append(feature.squeeze())

    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")
    
    return features

features = extract_features_tf_grayscale(r"C:\Users\Lenovo\Downloads\UCSD_Anomaly_Dataset.tar.gz\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001")
np.save("features_test_Test001.npy", features)
print(f"Features shape: {features.shape}")
