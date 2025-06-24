import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Klasifikasi Citra X-Ray", layout="wide")

# ================== INISIALISASI STATE ==================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Beranda"

# ===================== SIDEBAR ===========================
st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #fdfdfe;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem;
    }
    section[data-testid="stSidebar"] .stRadio label {
        background-color: #f9dce1;
        color: #884c5f;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 4px;
        display: block;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #f9cfd9;
        color: #4d2a33;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📋 Navigasi")
nav_options = ["🏠 Beranda", "🔍 Klasifikasi", "📊 Visualisasi", "ℹ️ Tentang"]
selected = st.sidebar.radio("", nav_options, index=nav_options.index("🏠 Beranda") if st.session_state["active_page"] == "Beranda" else nav_options.index(f"🔍 Klasifikasi" if st.session_state["active_page"] == "Klasifikasi" else st.session_state["active_page"]))
st.session_state["active_page"] = selected

# ===================== KONTEN ===========================
if selected.startswith("🏠"):
    st.markdown("<h1 style='color:#884c5f;'>Aplikasi Klasifikasi Citra Chest X-Ray</h1>", unsafe_allow_html=True)
    st.write("""
        Selamat datang! Aplikasi ini membantu mengklasifikasikan kondisi paru-paru dari citra X-Ray
        menjadi: **COVID-19**, **Normal**, atau **Pneumonia** menggunakan Deep Learning.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Normal_AP_CXR.jpg",
             caption="Contoh Citra Chest X-Ray", width=400)

elif selected.startswith("🔍"):
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

elif selected.startswith("📊"):
    st.markdown("<h1 style='color:#884c5f;'>Visualisasi Evaluasi Model</h1>", unsafe_allow_html=True)
    st.info("Fitur visualisasi seperti akurasi, confusion matrix, dan lainnya akan segera ditambahkan.")

elif selected.startswith("ℹ️"):
    st.markdown("<h1 style='color:#884c5f;'>Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan oleh **Luluatul Maknunah** untuk keperluan skripsi.

        **Tujuan:** Mengklasifikasikan citra X-Ray menjadi tiga kategori: COVID-19, Normal, Pneumonia.  
        **Model:** EfficientNet-B0 dan modifikasi dengan ECA.  
        **Pendamping:** Dosen Pembimbing Skripsi
    """)
