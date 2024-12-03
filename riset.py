import streamlit as st
import gdown
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Fungsi untuk mendownload model dari Google Drive
@st.cache_resource
def download_model():
    model_url = "https://drive.google.com/uc?id=1--XGx1vFU-VvlvPBed39Z40ehGv-Uttr"
    output_path = "best_model.keras"
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    return output_path

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize ke target size
    img_array = np.array(image)  # Convert ke numpy array
    if img_array.shape[-1] != 3:  # Pastikan ada 3 channel (RGB)
        raise ValueError("Gambar harus memiliki 3 channel (RGB).")
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# Judul aplikasi
st.title("Klasifikasi COVID-19, Pneumonia, dan Normal")
st.text("Aplikasi ini menggunakan model EfficientNet untuk mengklasifikasikan gambar X-Ray.")

# Unduh model
st.write("Mengunduh model dari Google Drive...")
model_path = download_model()

# Muat model
st.write("Memuat model...")
model = load_model(model_path)

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar X-Ray Anda", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    
    try:
        # Proses gambar
        image = Image.open(uploaded_file).convert("RGB")  # Pastikan gambar diubah ke RGB
        processed_image = preprocess_image(image)

        # Debugging bentuk input
        st.write(f"Bentuk input gambar: {processed_image.shape}")

        # Prediksi dengan model
        prediction = model.predict(processed_image)
        classes = ["COVID-19", "Pneumonia", "Normal"]
        predicted_class = classes[np.argmax(prediction)]
        
        # Tampilkan hasil prediksi
        st.write(f"Prediksi: **{predicted_class}**")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
