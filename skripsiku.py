import streamlit as st
import tensorflow as tf
from PIL import Image

# ==================== MENU ==================== #
menu = st.sidebar.selectbox("Pilih Menu", ["BERANDA", "KLASIFIKASI", "VISUALISASI", "TENTANG"])

# ==================== BERANDA ==================== #
if menu == "BERANDA":
    st.title("👩‍⚕️ Aplikasi Klasifikasi Citra Chest X-Ray")
    st.write("Selamat datang! Aplikasi ini digunakan untuk mendeteksi kondisi paru-paru berdasarkan gambar X-Ray menggunakan metode deep learning.")
    
    # Gunakan gambar dari URL atau hapus baris ini jika tidak punya file lokal
    #st.image("https://upload.wikimedia.org/wikipedia/commons/8/80/Normal_X-ray_image_of_human_lungs.jpg", 
           #  caption="Contoh Citra Chest X-Ray", use_column_width=True)

# ==================== KLASIFIKASI ==================== #
elif menu == "KLASIFIKASI":
    st.title("📥 Unggah Citra Chest X-Ray")

    # Upload gambar (hanya upload dan preview, tidak klasifikasi realtime)
    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar berhasil diunggah", use_column_width=True)
        st.info("Gambar berhasil ditampilkan. Proses klasifikasi akan dilakukan nanti oleh admin atau sistem batch.")

# ==================== VISUALISASI ==================== #
elif menu == "VISUALISASI":
    st.title("📊 VISUALISASI")
    st.write("Halaman ini dapat menampilkan visualisasi hasil evaluasi model atau data.")
    st.info("Fitur ini masih dalam pengembangan.")

# ==================== TENTANG ==================== #
elif menu == "TENTANG":
    st.title("📌 TENTANG APLIKASI")
    st.write("""
    Aplikasi ini dikembangkan untuk keperluan skripsi dalam bidang deteksi penyakit paru menggunakan citra Chest X-Ray.

    **Fitur:**
    - Upload dan tampilan citra Chest X-Ray.
    - Klasifikasi menggunakan EfficientNet-B0 dan modifikasi ECA (diproses secara terpisah).
    - Visualisasi hasil evaluasi model.
    
    **Pengembang:** Luluatul Maknunah  
    **Pembimbing:** Dosen Pembimbing Skripsi
    """)
