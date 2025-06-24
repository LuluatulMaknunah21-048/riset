import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# ========================= KONFIGURASI =========================
st.set_page_config(page_title="Klasifikasi Chest X-Ray", layout="wide", page_icon="🩻")
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stSelectbox, .stFileUploader, .stTextInput {
            background-color: white !important;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        .title-style {
            font-size: 32px;
            font-weight: 700;
            color: #4B8BBE;
        }
    </style>
""", unsafe_allow_html=True)

# ========================= MENU =========================
menu = st.sidebar.radio("📁 Navigasi", ["🏠 Beranda", "🧠 Klasifikasi", "📊 Visualisasi", "ℹ️ Tentang"])

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
if menu == "🏠 Beranda":
    st.markdown('<h1 class="title-style">Aplikasi Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)
    st.write("""
        Selamat datang di aplikasi klasifikasi citra X-Ray dada!  
        Aplikasi ini memanfaatkan teknologi Deep Learning untuk mendeteksi kondisi paru-paru dari citra X-Ray, seperti **COVID-19**, **Pneumonia**, dan **Normal**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

# ======================= KLASIFIKASI ======================
elif menu == "🧠 Klasifikasi":
    st.markdown('<h1 class="title-style">Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)

    # Pilih model
    st.subheader("📌 Pilih Model Klasifikasi")
    model_choice = st.selectbox("Model yang digunakan", ["EfficientNet-B0", "EfficientNet-B0 + ECA"])
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
        file_id = "1MYnpCeIOCDrq_4lkkWL9oTEmSeJPU60x"
    else:
        model_path = "model_efficientnetB0ECA.keras"
        file_id = "1xF39Jx-fNboR-3hGH5b8-kritCr-a-PG"

    model = load_model(model_path, file_id)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    # Upload gambar
    st.subheader("📤 Unggah Gambar")
    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="🖼️ Gambar yang diunggah", width=300)

        st.markdown("---")
        if st.button("🔍 Prediksi Sekarang"):
            with st.spinner("🧠 Memproses gambar..."):
                img = image.resize((224, 224))
                img_tensor = tf.convert_to_tensor(img)
                img_tensor = tf.expand_dims(img_tensor, axis=0)

                # Prediksi
                probs = model.predict(img_tensor)[0]
                pred_index = tf.argmax(probs).numpy()
                pred_label = class_labels[pred_index]

                st.success(f"✅ **Hasil Prediksi: {pred_label}**")
                st.subheader("📈 Probabilitas per Kelas:")
                for i, label in enumerate(class_labels):
                    st.markdown(f"- **{label}**: {probs[i]*100:.2f}%")

# ====================== VISUALISASI ======================
elif menu == "📊 Visualisasi":
    st.markdown('<h1 class="title-style">Visualisasi Evaluasi Model</h1>', unsafe_allow_html=True)
    st.info("📌 Fitur visualisasi evaluasi model akan tersedia pada versi berikutnya.")
    st.write("Rencana fitur: grafik akurasi, confusion matrix, visualisasi aktivasi layer, dan lainnya.")

# ======================== TENTANG ========================
elif menu == "ℹ️ Tentang":
    st.markdown('<h1 class="title-style">Tentang Aplikasi</h1>', unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan sebagai bagian dari penelitian klasifikasi penyakit paru-paru berbasis **Deep Learning**.

        ### 🔍 Fitur Utama:
        - Klasifikasi citra **COVID-19**, **Normal**, dan **Pneumonia**
        - Pilihan model: **EfficientNet-B0** dan **EfficientNet-B0 + ECA**
        - Antarmuka interaktif dan estetis

        ### 👤 Pengembang:
        **Luluatul Maknunah**  
        Dibimbing oleh: **Dosen Pembimbing Skripsi**

        Sumber data dan model berasal dari repositori publik dan penelitian internal.
    """)
