import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Klasifikasi Citra X-Ray", layout="wide")

# ================== INISIALISASI STATE ==================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Beranda"

# ===================== STYLING CSS ======================
st.markdown("""
    <style>
    body {
        background-color: #fdfdfe;
    }
    .nav-wrapper {
        position: sticky;
        top: 0;
        background-color: #fff;
        padding: 16px 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        z-index: 999;
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .nav-button {
        background-color: #fcefee;
        color: #884c5f;
        border: 1px solid #e7cbd2;
        padding: 10px 24px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        text-decoration: none;
    }
    .nav-button:hover {
        background-color: #f9dce1;
        color: #63313f;
    }
    .nav-selected {
        background-color: #f9cfd9 !important;
        color: #4d2a33 !important;
        border: 1px solid #e8a5b6 !important;
        font-weight: 700;
    }
    a.nav-button {
        text-decoration: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== NAVBAR ===========================
nav_options = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]
navbar_html = '<div class="nav-wrapper">'
for page in nav_options:
    selected_class = "nav-button nav-selected" if st.session_state["active_page"] == page else "nav-button"
    navbar_html += f'<a href="?page={page}" class="{selected_class}">{page}</a>'
navbar_html += '</div>'
st.markdown(navbar_html, unsafe_allow_html=True)

# ===================== HANDLE URL QUERY =================
params = st.query_params
if "page" in params:
    if params["page"] in nav_options:
        st.session_state["active_page"] = params["page"]

# ===================== KONTEN ===========================
selected = st.session_state["active_page"]

if selected == "Beranda":
    st.markdown("<h1 style='color:#884c5f;'>Aplikasi Klasifikasi Citra Chest X-Ray</h1>", unsafe_allow_html=True)
    st.write("""
        Selamat datang! Aplikasi ini membantu mengklasifikasikan kondisi paru-paru dari citra X-Ray
        menjadi: **COVID-19**, **Normal**, atau **Pneumonia** menggunakan Deep Learning.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg",
             caption="Contoh Citra Chest X-Ray", width=400)

elif selected == "Klasifikasi":
    st.markdown("<h1 style='color:#884c5f;'>Klasifikasi Citra X-Ray</h1>", unsafe_allow_html=True)

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
    st.markdown("<h1 style='color:#884c5f;'>Visualisasi Evaluasi Model</h1>", unsafe_allow_html=True)
    st.info("Fitur visualisasi seperti akurasi, confusion matrix, dan lainnya akan segera ditambahkan.")

elif selected == "Tentang":
    st.markdown("<h1 style='color:#884c5f;'>Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan oleh **Luluatul Maknunah** untuk keperluan skripsi.

        **Tujuan:** Mengklasifikasikan citra X-Ray menjadi tiga kategori: COVID-19, Normal, Pneumonia.  
        **Model:** EfficientNet-B0 dan modifikasi dengan ECA.  
        **Pendamping:** Dosen Pembimbing Skripsi
    """)
