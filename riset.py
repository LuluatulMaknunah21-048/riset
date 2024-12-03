import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Fungsi untuk memuat dan memproses gambar input (hanya resizing)
def load_and_process_image(img):
    img = img.resize((224, 224))  # Sesuaikan ukuran gambar dengan input model
    img_array = np.array(img)  # Ubah gambar menjadi array NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
    return img_array

# Fungsi untuk memuat model yang sudah dilatih
@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Menyiapkan aplikasi Streamlit
st.title("Klasifikasi X-ray: COVID-19, Pneumonia, atau Normal")
st.write("Aplikasi ini dapat mengklasifikasikan gambar X-ray menjadi salah satu dari tiga kelas: COVID-19, Pneumonia, atau Normal.")

# Upload gambar X-ray
uploaded_image = st.file_uploader("Pilih gambar X-ray untuk diklasifikasikan", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Menampilkan gambar yang diupload
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar X-ray yang diupload", use_column_width=True)
    
    # Memproses gambar untuk prediksi (hanya resizing)
    processed_image = load_and_process_image(img)
    
    # Memuat model yang sudah dilatih
    model_path = 'best_eff_adam.keras'  # Ganti dengan path model Anda
    model = load_trained_model(model_path)

    # Melakukan prediksi
    prediction = model.predict(processed_image)
    
    # Menampilkan hasil prediksi
    class_names = ['COVID-19', 'Pneumonia', 'Normal']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediksi: {predicted_class}")
    st.write(f"Probabilitas: {np.max(prediction) * 100:.2f}%")
