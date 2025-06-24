# install dulu di terminal kalau belum:
# pip install streamlit-option-menu

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

st.set_page_config(page_title="X-Ray Classifier", layout="wide")

# ======== Gaya Sidebar jadi Navbar Horizontal (atas) ======== #
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["🏠 Beranda", "🩻 Klasifikasi", "📊 Visualisasi", "ℹ️ Tentang"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f7f7f7"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "margin": "5px",
                "--hover-color": "#d6e4f0",
            },
            "nav-link-selected": {
                "background-color": "#a4c2f4", "color": "black"
            },
        },
    )

# ========== BERANDA ==========
if selected == "🏠 Beranda":
    st.title("👩‍⚕️ Aplikasi Klasifikasi Citra Chest X-Ray")
    st.markdown("Selamat datang! Deteksi otomatis kondisi paru-paru berbasis Deep Learning.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/80/Normal_X-ray_image_of_human_lungs.jpg", 
             caption="Contoh Citra Chest X-Ray", use_container_width=True)

# ========== KLASIFIKASI ==========
elif selected == "🩻 Klasifikasi":
    st.title("📥 Unggah Citra Chest X-Ray")

    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar berhasil diunggah", use_container_width=True)
        st.info("Gambar berhasil ditampilkan. Proses klasifikasi akan dilakukan nanti oleh sistem.")

# ========== VISUALISASI ==========
elif selected == "📊 Visualisasi":
    st.title("📊 Visualisasi Model dan Data")
    st.write("Halaman ini akan menampilkan metrik evaluasi dan visualisasi performa model.")
    st.warning("Fitur ini masih dalam pengembangan.")

# ========== TENTANG ==========
elif selected == "ℹ️ Tentang":
    st.title("🧾 Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan sebagai bagian dari penelitian klasifikasi penyakit paru berdasarkan gambar X-Ray.
    
    **Fitur Utama:**
    - Upload & klasifikasi gambar X-Ray paru-paru
    - Model: EfficientNet-B0 & ECA-modified
    - Visualisasi metrik evaluasi (segera hadir)

    **Pengembang:** Luluatul Maknunah 💫  
    **Pembimbing:** Dosen Pembimbing Skripsi  
    """)

