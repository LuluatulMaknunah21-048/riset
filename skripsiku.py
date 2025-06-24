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
        options=["Beranda", "Klasifikasi", "Visualisasi", "Tentang"],
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
if selected == "Beranda":
    st.title("Aplikasi Klasifikasi Citra Chest X-Ray")
    st.markdown("Selamat datang! Deteksi otomatis kondisi paru-paru berbasis Deep Learning.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/80/Normal_X-ray_image_of_human_lungs.jpg", 
             caption="Contoh Citra Chest X-Ray", use_container_width=True)

elif selected == "Klasifikasi":
    st.title("Klasifikasi Citra Chest X-Ray")

    # Pilihan model
    model_choice = st.selectbox("Pilih Metode Klasifikasi", ["EfficientNet-B0", "EfficientNet-ECA"])

    # Load model sesuai pilihan
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
    else:
        model_path = "effeca_model.keras"

    @st.cache_resource
    def load_model(path):
        return tf.keras.models.load_model(path)

    model = load_model(model_path)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        # Tampilkan gambar di tengah dengan ukuran kecil
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Gambar yang diunggah", width=300)

        # Tombol prediksi
        st.subheader("Prediksi Citra")
        if st.button("Prediksi"):
            with st.spinner("Memproses citra..."):
                # Siapkan input
                img_resized = image.resize((224, 224))
                img_tensor = tf.convert_to_tensor(img_resized)
                img_tensor = tf.expand_dims(img_tensor, axis=0)  # shape (1, 224, 224, 3)

                # Prediksi
                probs = model.predict(img_tensor)[0]
                pred_index = tf.argmax(probs).numpy()
                pred_label = class_labels[pred_index]

                # Tampilkan hasil
                st.success(f"Hasil Prediksi: {pred_label}")
                st.markdown("**Probabilitas per kelas:**")
                for i, label in enumerate(class_labels):
                    st.write(f"- {label}: {probs[i]*100:.2f}%")



# ========== VISUALISASI ==========
elif selected == "Visualisasi":
    st.title("📊 Visualisasi Model dan Data")
    st.write("Halaman ini akan menampilkan metrik evaluasi dan visualisasi performa model.")
    st.warning("Fitur ini masih dalam pengembangan.")

# ========== TENTANG ==========
elif selected == "Tentang":
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

