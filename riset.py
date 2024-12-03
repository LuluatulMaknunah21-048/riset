import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Fungsi untuk mengunduh model dari Google Drive
@st.cache_resource
def download_model_from_gdrive(file_id, output):
    """
    Mengunduh file model dari Google Drive menggunakan ID file.
    """
    if not os.path.exists(output):  # Cek apakah file sudah ada
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    return output

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(224, 224)):
    """
    Memproses gambar untuk prediksi model.
    """
    image = image.resize(target_size)  # Resize gambar ke ukuran target
    image_array = np.array(image)  # Ubah menjadi array NumPy
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    return image_array / 255.0  # Normalisasi piksel ke [0, 1]

# ID file Google Drive (ganti dengan ID Anda)
FILE_ID = "1--XGx1vFU-VvlvPBed39Z40ehGv-Uttr"  # ID dari URL Google Drive
MODEL_PATH = "best_model.keras"  # Nama file lokal untuk menyimpan model

# Unduh model
st.write("Mengunduh model dari Google Drive...")
model_file = download_model_from_gdrive(FILE_ID, MODEL_PATH)

# Memuat model
st.write("Memuat model...")
model = load_model(model_file)
st.success("Model berhasil dimuat!")

# Streamlit interface
st.title("Klasifikasi X-ray: COVID-19, Pneumonia, Normal")
uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar X-ray yang diunggah", use_column_width=True)

    # Memproses gambar
    processed_image = preprocess_image(image)

    # Prediksi menggunakan model
    prediction = model.predict(processed_image)
    class_names = ["COVID-19", "Pneumonia", "Normal"]
    predicted_class = class_names[np.argmax(prediction)]

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Kelas: {predicted_class}")
    st.write("Probabilitas untuk setiap kelas:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]:.2f}")
