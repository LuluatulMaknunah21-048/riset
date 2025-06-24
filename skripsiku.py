import streamlit as st
import tensorflow as tf
from PIL import Image

# ==================== MENU ==================== #
menu = st.sidebar.selectbox("Pilih Menu", ["BERANDA", "KLASIFIKASI", "VISUALISASI", "TENTANG"])

# ==================== BERANDA ==================== #
if menu == "BERANDA":
    st.title("👩‍⚕️ Aplikasi Klasifikasi Citra Chest X-Ray")
    st.write("Selamat datang! Aplikasi ini digunakan untuk mendeteksi kondisi paru-paru berdasarkan gambar X-Ray menggunakan metode deep learning.")
    st.image("chest_xray_sample.jpg", caption="Contoh Citra Chest X-Ray", use_column_width=True)

# ==================== KLASIFIKASI ==================== #
elif menu == "KLASIFIKASI":
    st.title("🔍 KLASIFIKASI CITRA CHEST X-RAY")

    # Pilih model
    model_choice = st.selectbox("Pilih Metode Klasifikasi", ["EfficientNet-B0", "EfficientNet-ECA"])

    # Load model dari file .keras
    if model_choice == "EfficientNet-B0":
        model_path = "effb0_model.keras"
    else:
        model_path = "effeca_model.keras"

    @st.cache_resource
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    model = load_model(model_path)
    class_labels = ['COVID-19', 'Normal', 'Pneumonia']

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar Chest X-Ray (.png/.jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Resize gambar
        img_resized = image.resize((224, 224))
        img_tensor = tf.convert_to_tensor(img_resized)
        img_tensor = tf.expand_dims(img_tensor, axis=0)  # shape (1, 224, 224, 3)

        if st.button("Prediksi"):
            with st.spinner("⏳ Sedang memproses..."):
                probs = model.predict(img_tensor)[0]
                pred_index = tf.argmax(probs).numpy()
                pred_label = class_labels[pred_index]

                st.success(f"Hasil Prediksi: **{pred_label}**")
                st.write("Probabilitas:")
                for i, label in enumerate(class_labels):
                    st.write(f"- {label}: {probs[i]*100:.2f}%")

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
    - Klasifikasi citra menggunakan EfficientNet-B0 dan versi modifikasi ECA.
    - Prediksi real-time dengan upload gambar.
    - Visualisasi dan evaluasi performa (akan ditambahkan).
    
    **Pengembang:** Luluatul Maknunah  
    **Pembimbing:** Dosen Pembimbing Skripsi  
    """)
