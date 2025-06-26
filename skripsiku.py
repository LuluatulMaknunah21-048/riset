import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown
import os
import numpy as np

st.set_page_config(page_title="Klasifikasi Citra X-Ray", layout="wide")

# ================== INISIALISASI STATE ==================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Beranda"

# ===================== STYLING ==================
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #884c5f;
    }
    div[class^="stButton"] > button {
        background-color: #f9dce1;
        color: #884c5f;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        width: 100%;
        text-align: left;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    div[class^="stButton"] > button:hover {
        background-color: #f9cfd9;
    }
    .active {
        background-color: #f7bcc9 !important;
        font-weight: 700;
        color: #4d2a33 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== MENU SIDEBAR ==================
st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
menu_list = ["Beranda", "Klasifikasi", "Visualisasi", "Tentang"]

for item in menu_list:
    if st.sidebar.button(item, key=item):
        st.session_state["active_page"] = item

selected = st.session_state["active_page"]

# ===================== KONTEN TIAP HALAMAN ===========================
if selected == "Beranda":
    st.markdown("<h1 style='color:#884c5f;'>Aplikasi Klasifikasi Citra Chest X-Ray</h1>", unsafe_allow_html=True)
    
    st.image("https://img.freepik.com/premium-vector/lungs-waving-hand-character_464314-5436.jpg",
             width=400)
    st.write("""
    Aplikasi ini dirancang untuk melakukan **klasifikasi** citra **X-Ray dada** menggunakan teknologi **Deep Learning**. 
    Model yang digunakan berbasis arsitektur **EfficientNet-B0** serta versi modifikasinya, yaitu **EfficientNet-ECA**.
    
    Model akan mengklasifikasikan citra ke dalam salah satu dari tiga kategori berikut: 
    **COVID-19**, **Normal**, atau **Pneumonia**.
    
    Silakan unggah citra X-Ray Anda pada menu klasifikasi dan lihat hasil klasifikasinya secara langsung!
    """)

    
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
            img = image.resize((224, 224)).Convert('RGB')
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            probs = model.predict(img_array)[0]
            #st.write(probs)
            pred_index = np.argmax(probs)
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
        Oleh **Luluatul Maknunah** NIM 210411100048 
        
        **Dosen Pembimbing 1 :** Prof. Dr. Rima Tri Wahyuningrum, S.T., M.T
        
        **Dosen Pembimbing 2 :** Dr. Cucun Very Angkoso, S.T., M.T
    """)
