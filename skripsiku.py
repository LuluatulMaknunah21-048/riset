import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Klasifikasi Chest X-Ray", layout="wide")

# ========================= MENU =========================
menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"])

# =================== FUNGSI DOWNLOAD MODEL ===============
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

# =================== FUNGSI LOAD MODEL ===================
@st.cache_resource
def load_model(model_path, file_id):
    download_model(file_id, model_path)
    return tf.keras.models.load_model(model_path)

# ======================== BERANDA ========================
if menu == "Beranda":
    st.title("Aplikasi Klasifikasi Citra Chest X-Ray")
    st.write("""
        Aplikasi ini dirancang untuk mengklasifikasikan kondisi paru-paru berdasarkan citra X-Ray
        menggunakan model deep learning.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

# ======================= KLASIFIKASI ======================
elif menu == "Klasifikasi":
    st.title("Klasifikasi Citra Chest X-Ray")

    # Pilih model
    model_choice = st.selectbox("Pilih Model Klasifikasi", ["EfficientNet-B0", "EfficientNet-B0 + ECA"])
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
        file_id = "1MYnpCeIOCDrq_4lkkWL9oTEmSeJPU60x"
    else:
        model_path = "model_efficientnetB0ECA.keras"
        file_id = "1xF39Jx-fNboR-3hGH5b8-kritCr-a-PG"

    model = load_model(model_path, file_id)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Gambar yang diunggah", width=300)

        if st.button("Prediksi"):
            with st.spinner("Memproses gambar..."):
                # Siapkan input
                img = image.resize((224, 224))
                img_tensor = tf.convert_to_tensor(img)
                img_tensor = tf.expand_dims(img_tensor, axis=0)

                # Prediksi
                probs = model.predict(img_tensor)[0]
                pred_index = tf.argmax(probs).numpy()
                pred_label = class_labels[pred_index]

                st.success(f"Hasil Prediksi: {pred_label}")
                st.markdown("**Probabilitas per kelas:**")
                for i, label in enumerate(class_labels):
                    st.write(f"- {label}: {probs[i]*100:.2f}%")

# ====================== VISUALISASI ======================
elif menu == "Visualisasi":
    st.title("Visualisasi Evaluasi Model")
    st.info("Fitur visualisasi evaluasi model akan ditambahkan pada versi berikutnya.")
    st.write("Anda dapat menambahkan grafik akurasi, confusion matrix, atau visualisasi aktivasi layer di sini.")

# ======================== TENTANG ========================
elif menu == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("""
        Aplikasi ini dikembangkan untuk mendeteksi penyakit paru berdasarkan citra X-Ray
        menggunakan dua model klasifikasi: EfficientNet-B0 dan versi modifikasi dengan ECA (Efficient Channel Attention).

        **Fitur Utama:**
        - Klasifikasi citra COVID-19, Normal, dan Pneumonia
        - Pilihan dua arsitektur model
        - UI interaktif dan profesional

        **Pengembang:** Luluatul Maknunah  
        **Pembimbing:** Dosen Pembimbing Skripsi
    """)
