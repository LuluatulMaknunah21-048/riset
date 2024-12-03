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
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# Sidebar untuk navigasi
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Home", "Petunjuk", "Tentang"])

if app_mode == "Home":
    # Judul aplikasi
    st.title("KLASIFIKASI CITRA CHEST X-RAY MENGGUNAKAN KOMBINASI EFFICIENTNET DAN EFFICIENT CHANNEL ATTENTION (ECA) ")
    st.text("Aplikasi ini menggunakan arsitektur EfficientNet-B0.")

    # Unduh model
    #st.write("Mengunduh model dari Google Drive...")
    model_path = download_model()

    # Muat model
    #st.write("Memuat model...")
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

            # Label yang sesuai dengan kelas yang digunakan saat pelatihan (berurutan secara alfabet)
            labels = ['COVID-19', 'Normal', 'Pneumonia']  # Urutan berdasarkan alfabet

            # Prediksi dengan model (output model berupa probabilitas untuk setiap kelas)
            prediction = model.predict(processed_image)

            # Ambil indeks kelas dengan probabilitas tertinggi
            predicted_class_index = np.argmax(prediction)

            # Menampilkan nama kelas sesuai dengan label
            predicted_class = labels[predicted_class_index]

            # Menampilkan hasil prediksi di Streamlit
            st.write(f"Prediksi: **{predicted_class}**")
            st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

elif app_mode == "Petunjuk":
    st.title("Petunjuk Penggunaan")
    st.write("""
        1. Pilih menu **Home** untuk mengunggah Citra Chest X-Ray.
        2. Pilih Citra Chest X-Ray yang ingin Anda klasifikasikan.
        3. Aplikasi ini akan memberikan prediksi apakah gambar tersebut termasuk **COVID-19**, **Pneumonia**, atau **Normal**.
        4. Hasil prediksi akan menunjukkan nama kelas dan tingkat kepercayaan model.
    """)

elif app_mode == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("""
        Aplikasi ini menggunakan model **EfficientNetB0** yang telah dilatih untuk mengklasifikasikan Chest X-Ray menjadi tiga kelas:
        - **COVID-19**
        - **Pneumonia**
        - **Normal**
    """)
    st.write("""
        oleh : <br>
        Lu'luatul Maknunah 210411100048 <br>
        Dosen Pembimbing Riset : Prof. Dr. Rima Tri Wahyuningrum, S.T., MT.
    """, unsafe_allow_html=True)
