import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# ====================== KONFIGURASI DASAR ======================
st.set_page_config(page_title="Klasifikasi Chest X-Ray", layout="wide", page_icon="🩻")

# ====================== GAYA TAMBAHAN (CSS) =====================
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
            padding: 0.5em 1em;
        }
        .stSelectbox, .stFileUploader {
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

# =================== MENU NAVIGASI TANPA REFRESH ==================
menu_options = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Beranda"

selected = st.sidebar.radio("Menu", menu_options, index=menu_options.index(st.session_state.selected_menu))
st.session_state.selected_menu = selected

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
if selected == "Beranda":
    st.markdown('<h1 class="title-style">Aplikasi Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)
    st.write("""
        Selamat datang!  
        Aplikasi ini menggunakan Deep Learning untuk mengklasifikasikan kondisi paru-paru dari citra X-Ray: **COVID-19**, **Normal**, dan **Pneumonia**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

# ======================= KLASIFIKASI ======================
elif selected == "Klasifikasi":
    st.markdown('<h1 class="title-style">Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)

    model_choice = st.selectbox("Pilih Model Klasifikasi", ["EfficientNet-B0", "EfficientNet-B0 + ECA"])
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
        file_id = "1MYnpCeIOCDrq_4lkkWL9oTEmSeJPU60x"
    else:
        model_path = "model_efficientnetB0ECA.keras"
        file_id = "1xF39Jx-fNboR-3hGH5b8-kritCr-a-PG"

    model = load_model(model_path, file_id)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    st.subheader("Unggah Gambar X-Ray")
    uploaded_file = st.file_uploader("Pilih file gambar (.png/.
