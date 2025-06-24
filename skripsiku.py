import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Citra X-Ray", layout="wide")

# ======= Simpan status menu aktif ========
if "active_page" not in st.session_state:
    st.session_state.active_page = "Beranda"

# ======= Styling Navbar Pastel Estetik ========
st.markdown("""
    <style>
    /* Global */
    body {
        background-color: #fdfdfe;
    }

    /* Navbar container */
    .navbar {
        background-color: #fcefee;
        padding: 12px 24px;
        border-radius: 12px;
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-bottom: 30px;
        box-shadow: 0px 2px 6px rgba(200, 200, 200, 0.3);
    }

    /* Navbar link style */
    .nav-item {
        padding: 8px 20px;
        border-radius: 8px;
        background-color: #fcefee;
        color: #884c5f;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }

    /* Hover */
    .nav-item:hover {
        background-color: #f9dce1;
        color: #63313f;
    }

    /* Active */
    .active {
        background-color: #f9cfd9 !important;
        color: #4d2a33 !important;
        font-weight: 600;
        box-shadow: 0px 0px 0px 2px #f8bfcf;
    }

    .title-style {
        font-size: 30px;
        font-weight: 700;
        color: #884c5f;
    }
    </style>
""", unsafe_allow_html=True)

# ======= Navbar HTML ========
st.markdown('<div class="navbar">', unsafe_allow_html=True)

menu_items = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]
for item in menu_items:
    active = "nav-item active" if st.session_state.active_page == item else "nav-item"
    if st.markdown(f'<div class="{active}" onclick="window.location.href=\'#{item}\'">{item}</div>', unsafe_allow_html=True):
        st.session_state.active_page = item

st.markdown('</div>', unsafe_allow_html=True)

# ======= Menentukan halaman aktif (tanpa URL refresh) ========
selected = st.session_state.active_page

# ==================== Konten Halaman ======================
if selected == "Beranda":
    st.markdown('<h1 class="title-style">Aplikasi Klasifikasi Citra Chest X-Ray</h1>', unsafe_allow_html=True)
    st.write("Selamat datang! Aplikasi ini membantu mengklasifikasikan citra X-Ray paru-paru menjadi COVID-19, Normal, atau Pneumonia.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg", caption="Contoh Citra Chest X-Ray", width=400)

elif selected == "Klasifikasi":
    st.markdown('<h1 class="title-style">Klasifikasi X-Ray</h1>', unsafe_allow_html=True)

    model_choice = st.selectbox("Pilih Model", ["EfficientNet-B0", "EfficientNet-B0 + ECA"])
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
        file_id = "1MYnpCeIOCDrq_4lkkWL9oTEmSeJPU60x"
    else:
        model_path = "model_efficientnetB0ECA.keras"
        file_id = "1xF39Jx-fNboR-3hGH5b8-kritCr-a-PG"

    @st.cache_resource
    def load_model(model_path, file_id):
        if not os.path.exists(model_path):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
        return tf.keras.models.load_model(model_path)

    model = load_model(model_path, file_id)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    uploaded_file = st.file_uploader("Unggah Gambar Chest X-Ray", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", width=300)

        if st.button("Prediksi"):
            img = image.resize((224, 224))
            img_tensor = tf.convert_to_tensor(img)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            probs = model.predict(img_tensor)[0]
            pred_index = tf.argmax(probs).numpy()
            pred_label = class_labels[pred_index]

            st.success(f"Hasil Prediksi: **{pred_label}**")
            for i, label in enumerate(class_labels):
                st.write(f"{label}: {probs[i]*100:.2f}%")

elif selected == "Visualisasi":
    st.markdown('<h1 class="title-style">Visualisasi Evaluasi Model</h1>', unsafe_allow_html=True)
    st.info("Fitur ini akan menampilkan grafik evaluasi seperti akurasi, confusion matrix, dan lainnya.")

elif selected == "Tentang":
    st.markdown('<h1 class="title-style">Tentang Aplikasi</h1>', unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan oleh Luluatul Maknunah untuk skripsi klasifikasi penyakit paru berbasis citra X-Ray.
        
        - Model: EfficientNet-B0 & EfficientNet-B0 + ECA  
        - Data: Gambar X-Ray COVID-19, Normal, dan Pneumonia  
        - Tujuan: Membantu analisis radiografi medis dengan AI
    """)
