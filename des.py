import streamlit as st
import numpy as np
import pandas as pd
import os
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import pickle

# Load model ResNet50 pre-trained dengan GAP
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Fungsi untuk ekstraksi fitur dari satu gambar
def extract_features_from_image(img_path):
    try:
        # Load dan resize gambar
        img = load_img(img_path, target_size=(224, 224))  # Memuat gambar dan mengubah ukurannya menjadi 224x224 (ukuran input ResNet50).
        img_array = img_to_array(img)  # Ubah gambar ke array numpy
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

        # Ekstraksi fitur
        feature = model.predict(img_array).flatten()  # Vektor fitur 1D
        return feature

    except Exception as e:
        st.error(f"Error processing {img_path}: {e}")
        return None

# Main Streamlit App
st.title("Ekstraksi Fitur, PCA, dan Prediksi")

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load PCA model
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Load FFNN model
with open('ffnn_model.pkl', 'rb') as f:
    ffnn_model = pickle.load(f)

# Upload gambar melalui Streamlit
uploaded_file = st.file_uploader("Unggah gambar untuk prediksi", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded_file is not None:
    # Simpan gambar yang diunggah ke direktori sementara
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp_image.png", caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Ekstrak Fitur dan Prediksi"):
        st.write("Memulai ekstraksi fitur...")
        features = extract_features_from_image("temp_image.png")

        if features is not None:
            # Standarisasi fitur
            st.write("Standarisasi fitur...")
            standardized_features = scaler.transform([features])

            # Aplikasi PCA
            st.write("Mengaplikasikan PCA...")
            pca_features = pca.transform(standardized_features)

            # Prediksi menggunakan FFNN
            st.write("Melakukan prediksi...")
            prediction = ffnn_model.predict(pca_features)

            # Tampilkan hasil
            st.write("Hasil prediksi:", prediction[0])

st.write("\n\n---\n**Instruksi:**")
st.write("1. Unggah gambar dalam format yang didukung (PNG, JPG, JPEG, BMP, GIF).")
st.write("2. Klik tombol untuk melakukan ekstraksi fitur, PCA, dan prediksi.")
