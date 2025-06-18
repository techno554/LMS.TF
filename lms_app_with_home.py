import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="LMS Kelulusan", layout="centered")
st.sidebar.title("🏠 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Home", "Prediksi Kelulusan"])

if menu == "Home":
    st.title("🏫 Dashboard LMS - Kelulusan Peserta")
    uploaded_file = st.file_uploader("Unggah Dataset Peserta (Excel)", type=["xlsx"], key="home_upload")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.rename(columns={
            'Partisipasi': 'partisipasi_forum',
            'Kehadiran': 'total_login',
            'status_kelulusan': 'Status_Kelulusan'
        })
        df['partisipasi_forum'] = df['partisipasi_forum'].astype(str).str.replace('%', '').astype(float)
        df['total_login'] = df['total_login'].astype(float)
        df['Status_Kelulusan'] = df['Status_Kelulusan'].map({'Lulus': 1, 'Tidak Lulus': 0})

        st.subheader("📋 Daftar Peserta")
        st.dataframe(df)

        search = st.text_input("🔍 Cari Nama atau ID Peserta")
        if search:
            results = df[df['ID_Peserta'].str.contains(search, case=False) | df['Nama'].str.contains(search, case=False)]
            if not results.empty:
                for _, row in results.iterrows():
                    status = "✅ LULUS" if row['Status_Kelulusan'] == 1 else "❌ TIDAK LULUS"
                    st.write(f"**{row['ID_Peserta']} - {row['Nama']}**: {status}")
            else:
                st.warning("Peserta tidak ditemukan.")

elif menu == "Prediksi Kelulusan":

# Upload dataset
uploaded_file = st.file_uploader("Unggah Dataset Peserta (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Rename kolom agar sesuai dengan format model regresi
    df = df.rename(columns={
        'Partisipasi': 'partisipasi_forum',
        'Kehadiran': 'total_login',
        'status_kelulusan': 'Status_Kelulusan'
    })

    # Hapus tanda persen dan ubah ke float
    df['partisipasi_forum'] = df['partisipasi_forum'].astype(str).str.replace('%', '').astype(float)
    df['total_login'] = df['total_login'].astype(float)

    # Tambahkan kolom dummy untuk materi dan skor jika tidak ada
    if 'materi_selesai' not in df.columns:
        df['materi_selesai'] = 50
    if 'skor_kuis_rata2' not in df.columns:
        df['skor_kuis_rata2'] = 75.0
    if 'durasi_total_akses' not in df.columns:
        df['durasi_total_akses'] = 30.0

    # Encode target
    df['Status_Kelulusan'] = df['Status_Kelulusan'].map({'Lulus': 1, 'Tidak Lulus': 0})

    expected_columns = {'total_login', 'materi_selesai', 'skor_kuis_rata2', 'partisipasi_forum', 'durasi_total_akses', 'Status_Kelulusan'}
    if expected_columns.issubset(df.columns):
        st.subheader("📋 Pratinjau Data")
        st.dataframe(df)

        # Model
        model = LogisticRegression()
        X = df[['total_login', 'materi_selesai', 'skor_kuis_rata2', 'partisipasi_forum', 'durasi_total_akses']]
        y = df['Status_Kelulusan']
        model.fit(X, y)

        st.subheader("🔍 Prediksi Kelulusan Peserta Baru")
        login = st.number_input("Total Login", min_value=0, value=10)
        materi = st.number_input("Materi Selesai", min_value=0, value=50)
        skor = st.slider("Skor Kuis Rata-rata", 0.0, 100.0, 75.0)
        forum = st.number_input("Partisipasi Forum", min_value=0, value=10)
        durasi = st.number_input("Durasi Total Akses (menit)", min_value=0.0, value=30.0)

        if st.button("Prediksi Kelulusan"):
            data = np.array([[login, materi, skor, forum, durasi]])
            prob = model.predict_proba(data)[0][1]
            status = "LULUS" if prob >= 0.5 else "TIDAK LULUS"
            st.success(f"✅ Prediksi Status: **{status}** dengan probabilitas {prob:.4f}")

            st.subheader("📈 Probabilitas Prediksi")
            prob_lulus = float(prob)
            prob_tidak = 1 - prob_lulus
            prob_df = pd.DataFrame({
                'Status': ['LULUS', 'TIDAK LULUS'],
                'Probabilitas': [prob_lulus, prob_tidak]
            })
            st.bar_chart(prob_df.set_index('Status'))

            st.subheader("📦 Aturan Akses Modul")
            if prob >= 0.5:
                st.info("🔓 Akses DIBERIKAN ke modul lanjutan.")
            else:
                st.warning("🔒 Akses DITOLAK. Peserta belum memenuhi syarat kelulusan.")

            st.subheader("🔄 Flowchart Evaluasi")
            st.graphviz_chart("""
                digraph {
                    Input_Data -> Preprocessing -> Model_Regresi_Logistik -> Hitung_Probabilitas
                    Hitung_Probabilitas -> Tentukan_Status -> Tampilkan_Hasil
                }
            """)

        st.subheader("📁 Simulasi Dokumen & Metadata")
        dokumen = pd.DataFrame({
            'NamaDokumen': ['Modul Excel Dasar', 'Modul Excel Lanjutan'],
            'Kategori': ['modul', 'modul'],
            'Kata_Kunci': ['excel, dasar', 'excel, lanjutan, formula'],
            'Versi': ['1.0', '2.1'],
            'Tahun': [2023, 2024]
        })
        st.dataframe(dokumen)
    else:
        st.error("Dataset tidak sesuai. Wajib memiliki kolom: total_login, materi_selesai, skor_kuis_rata2, partisipasi_forum, durasi_total_akses, Status_Kelulusan")
else:
    st.info("Silakan unggah file Excel terlebih dahulu untuk mulai.")
