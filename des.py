import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import joblib  # Ganti pickle dengan joblib

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

# Load the FFNN model if it exists
if os.path.exists(ffnn_model_path):
    print("Loading FFNN model...")
    ffnn_model = tf.keras.models.load_model(ffnn_model_path)
else:
    print("FFNN model file is missing, cannot load the model.")

st.title("Ekstraksi Fitur Gambar dan Prediksi")

uploaded_file = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded_file is not None:
    # Simpan file yang diunggah sementara
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Tampilkan gambar yang diunggah
    st.image("temp_image.png", caption="Gambar yang diunggah", use_column_width=True)
    print(f"File yang diunggah disimpan sebagai temp_image.png")

    if os.path.exists("temp_image.png"):
        print("Gambar berhasil diunggah.")
        # Tombol untuk mengekstrak fitur dan melakukan prediksi
        if st.button("Ekstrak Fitur dan Prediksi"):
            features = extract_features_from_image("temp_image.png")
            if features is not None:
                st.write("Fitur berhasil diekstrak.")
                
                # Gunakan scaler dan PCA untuk memproses fitur
                scaled_features = scaler.transform([features])
                pca_features = pca.transform(scaled_features)
                
                # Tambahkan kode prediksi di sini jika diperlukan, misalnya:
                # prediction = some_model.predict(pca_features)
                # st.write(f"Prediksi: {prediction}")
                st.write("Fitur diproses dengan PCA dan scaling.")
