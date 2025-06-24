import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# ========================= KONFIGURASI =========================
st.set_page_config(page_title="Klasifikasi Chest X-Ray", layout="wide", page_icon="🩻")

# ======================= GAYA TAMBAHAN =========================
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
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .sidebar-menu {
            margin: 10px 0;
        }
        .sidebar-menu a {
            display: block;
            padding: 8px 16px;
            border-radius: 6px;
            color: #333;
            text-decoration: none;
            font-weight: 500;
        }
        .sidebar-menu a:hover {
            background-color: #e3eaf2;
            color: #007ACC;
        }
        .sidebar-menu .active {
            background-color: #d0e2ff;
            color: #004b9b;
        }
    </style>
""", unsafe_allow_html=True)

# ========================= SIDEBAR NAV =========================
st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)

menu_items = {
    "Beranda": "beranda",
    "Klasifikasi": "klasifikasi",
    "Visualisasi": "visualisasi",
    "Tentang": "tentang"
}

selected = st.session_state.get("selected_menu", "Beranda")

for label, key in menu_items.items():
    active = "active" if selected == label else ""
    if st.sidebar.markdown(f'<div class="sidebar-menu"><a href="?menu={key}" class="{active}">{label}</a></div>', unsafe_allow_html=True):
        selected = label
        st.session_state.selected_menu = selected

# Ambil parameter dari URL jika ada
query_params = st.experimental_get_query_params()
menu = query_params.get("menu", ["beranda"])[0]

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
if menu == "beranda":
    st.markdown('<h1 class="title-style">Aplikasi Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)
    st.write("""
        Selamat datang!  
        Aplikasi ini menggunakan Deep Learning untuk mengklasifikasikan kondisi paru-paru dari citra X-Ray: **COVID-19**, **Normal**, dan **Pneumonia**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

# ======================= KLASIFIKASI ======================
elif menu == "klasifikasi":
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
    uploaded_file = st.file_uploader("Pilih file gambar (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Gambar yang Diunggah", width=300)

        st.markdown("---")
        if st.button("Prediksi"):
            with st.spinner("Memproses gambar..."):
                img = image.resize((224, 224))
                img_tensor = tf.convert_to_tensor(img)
                img_tensor = tf.expand_dims(img_tensor, axis=0)

                probs = model.predict(img_tensor)[0]
                pred_index = tf.argmax(probs).numpy()
                pred_label = class_labels[pred_index]

                st.success(f"**Hasil Prediksi: {pred_label}**")
                st.subheader("Probabilitas per Kelas:")
                for i, label in enumerate(class_labels):
                    st.markdown(f"- **{label}**: {probs[i]*100:.2f}%")

# ====================== VISUALISASI ======================
elif menu == "visualisasi":
    st.markdown('<h1 class="title-style">Visualisasi Evaluasi Model</h1>', unsafe_allow_html=True)
    st.info("Fitur visualisasi seperti akurasi, confusion matrix, dan aktivasi layer akan tersedia di versi selanjutnya.")

# ======================== TENTANG ========================
elif menu == "tentang":
    st.markdown('<h1 class="title-style">Tentang Aplikasi</h1>', unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan untuk membantu dalam klasifikasi penyakit paru-paru berdasarkan citra X-Ray.

        ### Fitur:
        - Klasifikasi 3 kelas: COVID-19, Normal, Pneumonia
        - Dua model pilihan: EfficientNet-B0 dan EfficientNet-B0 + ECA
        - Desain modern dan interaktif

        ### Pengembang:
        **Luluatul Maknunah**  
        Pembimbing: **Dosen Pembimbing Skripsi**
    """)
