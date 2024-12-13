import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # Ganti pickle dengan joblib
import gdown

# Function to download file from Google Drive using gdown
def download_file_from_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {output_path} successfully.")
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")

# URLs of the files stored on Google Drive
scaler_url = 'https://drive.google.com/uc?id=1-b1sbOb0iPMVbFL7mRAxFokC4yKLt8N4'
pca_url = 'https://drive.google.com/uc?id=1-NviIjOFmDaldOtJbJLBRYO8vH1qGm5R'
ffnn_url = 'https://drive.google.com/uc?id=1-bh0Ce3ag9RIonvM8kkWTPDyUiXWthRA'

# Define the paths where the files will be saved locally
scaler_path = 'scalerWithpca.pkl'
pca_model_path = 'pca_best.pkl'
ffnn_model_path = 'ffnnWithpca.h5'

# Download the files
download_file_from_drive(scaler_url, scaler_path)
download_file_from_drive(pca_url, pca_model_path)
download_file_from_drive(ffnn_url, ffnn_model_path)

# Load the scaler
print("Loading scaler...")
scaler = joblib.load(scaler_path)

# Load PCA model
print("Loading PCA model...")
pca = joblib.load(pca_model_path)

# Load the FFNN model
print("Loading FFNN model...")
ffnn_model = tf.keras.models.load_model(ffnn_model_path)

# Load ResNet50 for feature extraction
print("Loading ResNet50 model for feature extraction...")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features using ResNet50
def extract_features_from_image(img_path):
    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Extract features using ResNet50
        features = resnet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

st.title("IMPLEMENTASI PRINCIPAL COMPONENT ANALYSIS UNTUK MODIFIKASI ARSITEKTUR RESNET50 PADA KLASIFIKASI COVID-19")

uploaded_file = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image("temp_image.png", caption="Gambar yang diunggah", use_container_width=True)

    if st.button("Ekstrak Fitur dan Prediksi"):
        # Extract features
        features = extract_features_from_image("temp_image.png")
        if features is not None:
            st.write("Fitur berhasil diekstrak.")

            # Scale and transform features
            scaled_features = scaler.transform([features])
            pca_features = pca.transform(scaled_features)

            # Predict using FFNN model
            prediction = ffnn_model.predict(pca_features)
            predicted_class = np.argmax(prediction, axis=1)
            
            st.write(f"Prediksi: {predicted_class[0]}")
