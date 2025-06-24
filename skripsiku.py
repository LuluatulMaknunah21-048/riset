import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os

# Inisialisasi state
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

# Gaya CSS sidebar
st.sidebar.markdown("""
    <style>
    .custom-sidebar .menu-item {
        padding: 10px 18px;
        margin-bottom: 6px;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 500;
        background-color: #f2f4f8;
        color: #333333;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .custom-sidebar .menu-item:hover {
        background-color: #e0e7ef;
        color: #0056b3;
    }
    .custom-sidebar .active {
        background-color: #d0e2ff;
        color: #003d80;
        border-left: 4px solid #2f80ed;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar HTML
st.sidebar.markdown('<div class="sidebar-title"><b>Menu</b></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="custom-sidebar">', unsafe_allow_html=True)

menu_items = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]

for item in menu_items:
    is_active = "active" if st.session_state.menu == item else ""
    if st.sidebar.markdown(
        f'<div class="menu-item {is_active}" onclick="window.location.href=\'#{item}\'">{item}</div>',
        unsafe_allow_html=True,
    ):
        st.session_state.menu = item

st.sidebar.markdown('</div>', unsafe_allow_html=True)
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
