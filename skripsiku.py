import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Klasifikasi Chest X-Ray", layout="wide", page_icon="🩻")

# ======================== GAYA ========================
# Atas kode Python-mu, tambahkan styling ini:
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .menu-button {
        display: block;
        width: 100%;
        text-align: left;
        padding: 8px 16px;
        margin-bottom: 8px;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 500;
        background-color: #f0f2f6;
        border: none;
        color: #333333;
        transition: all 0.2s ease;
    }
    .menu-button:hover {
        background-color: #e0e8f0;
        color: #0056b3;
    }
    .menu-active {
        background-color: #cfe2ff !important;
        color: #003d80 !important;
        font-weight: 600;
        border-left: 5px solid #297eff;
    }
    </style>
""", unsafe_allow_html=True)

# Gunakan ini untuk sidebar-mu:
st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)

menu_options = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

for label in menu_options:
    css_class = "menu-button"
    if st.session_state.menu == label:
        css_class += " menu-active"
    if st.sidebar.button(label, key=label):
        st.session_state.menu = label

# ====================== DOWNLOAD MODEL ===================
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

@st.cache_resource
def load_model(model_path, file_id):
    download_model(file_id, model_path)
    return tf.keras.models.load_model(model_path)

# =========================== BERANDA ==========================
if st.session_state.menu == "Beranda":
    st.markdown('<h1 style="color:#4B8BBE;">Aplikasi Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)
    st.write("""
        Selamat datang!  
        Aplikasi ini menggunakan Deep Learning untuk mengklasifikasikan kondisi paru-paru dari citra X-Ray: **COVID-19**, **Normal**, dan **Pneumonia**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

# ========================= KLASIFIKASI ========================
elif st.session_state.menu == "Klasifikasi":
    st.markdown('<h1 style="color:#4B8BBE;">Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)

    model_choice = st.selectbox("Pilih Model Klasifikasi", ["EfficientNet-B0", "EfficientNet-B0 + ECA"])
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
        file_id = "1MYnpCeIOCDrq_4lkkWL9oTEmSeJPU60x"
    else:
        model_path = "model_efficientnetB0ECA.keras"
        file_id = "1xF39Jx-fNboR-3hGH5b8-kritCr-a-PG"

    model = load_model(model_path, file_id)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])
    
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

# ====================== VISUALISASI ============================
elif st.session_state.menu == "Visualisasi":
    st.markdown('<h1 style="color:#4B8BBE;">Visualisasi Evaluasi Model</h1>', unsafe_allow_html=True)
    st.info("Fitur visualisasi seperti akurasi, confusion matrix, dan aktivasi layer akan tersedia di versi selanjutnya.")

# =========================== TENTANG ============================
elif st.session_state.menu == "Tentang":
    st.markdown('<h1 style="color:#4B8BBE;">Tentang Aplikasi</h1>', unsafe_allow_html=True)
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
