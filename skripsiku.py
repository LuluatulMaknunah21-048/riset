import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# ===================== KONFIGURASI =====================
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
    .navbar-container {
        display: flex;
        justify-content: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: white;
        padding: 12px 0 8px 0;
        box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    .nav-button button {
        background-color: #ffffff;
        color: #884c5f;
        border: 1px solid #e7cbd2;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 500;
        padding: 6px 18px;
        margin: 0 6px;
        transition: 0.2s;
    }
    .nav-button button:hover {
        background-color: #fae6ec;
        border-color: #eabac6;
    }
    .nav-button button.selected {
        background-color: #f9cfd9;
        color: #4d2a33;
        border-color: #e8a5b6;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== NAVBAR ===========================
st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
nav_options = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]
nav_cols = st.columns(len(nav_options))
for i, option in enumerate(nav_options):
    is_selected = st.session_state["active_page"] == option
    btn_key = f"nav_{option}"
    with nav_cols[i]:
        if st.button(option, key=btn_key):
            st.session_state["active_page"] = option
        if is_selected:
            st.markdown(f"""
                <style>
                div[data-testid="stButton"][key="{btn_key}"] button {{
                    background-color: #f9cfd9;
                    color: #4d2a33;
                    border-color: #e8a5b6;
                    font-weight: 600;
                }}
                </style>
            """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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
