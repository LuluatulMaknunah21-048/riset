import streamlit as st

# Inisialisasi state menu jika belum ada
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

# Tampilan Header Menu Horizontal
st.markdown("<h2 style='text-align: center;'>Aplikasi Klasifikasi Chest X-Ray</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Beranda"):
        st.session_state.menu = "Beranda"
with col2:
    if st.button("Klasifikasi"):
        st.session_state.menu = "Klasifikasi"
with col3:
    if st.button("Visualisasi"):
        st.session_state.menu = "Visualisasi"
with col4:
    if st.button("Tentang"):
        st.session_state.menu = "Tentang"

# Ambil menu aktif
menu = st.session_state.menu

# ================= KONTEN UTAMA =================
if menu == "Beranda":
    st.subheader("Beranda")
    st.write("Selamat datang di aplikasi klasifikasi penyakit paru berdasarkan citra Chest X-Ray.")

elif menu == "Klasifikasi":
    st.subheader("Klasifikasi Citra")
    st.write("Silakan unggah gambar untuk dilakukan klasifikasi.")

elif menu == "Visualisasi":
    st.subheader("Visualisasi Hasil")
    st.info("Fitur visualisasi masih dalam pengembangan.")

elif menu == "Tentang":
    st.subheader("Tentang Aplikasi")
    st.write("Aplikasi ini dikembangkan untuk mendeteksi COVID-19, Pneumonia, dan kondisi Normal dari gambar X-Ray.")

