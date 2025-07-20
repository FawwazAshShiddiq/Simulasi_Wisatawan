import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import plotly.express as px

# layout 
st.set_page_config(layout="wide")

# JUDUL DAN PENJELASAN APLIKASI
st.title("📈 Program Simulasi Monte Carlo Kunjungan Wisatawan")
st.write("""
Aplikasi ini melakukan simulasi Monte Carlo untuk memprediksi jumlah kunjungan wisatawan di masa depan.
Aplikasi ini sudah disesuaikan untuk membaca format file CSV Anda.
""")


# PENTING: Konfigurasi Nama Kolom

KOLOM_NEGARA = 'negara'
KOLOM_PENGUNJUNG = 'jumlah_pengunjung'


# BAGIAN 1: UPLOAD FILE
with st.sidebar:
    st.header("Unggah File Data")
    uploaded_files = st.file_uploader(
        "Unggah 3 file CSV Anda di sini (misal: data_2022.csv, data_2023.csv, dst.)",
        type="csv",
        accept_multiple_files=True
    )
    st.info("""
    **Catatan:** Pastikan nama file mengandung tahun (misalnya '2024') agar program bisa mendeteksi tahun secara otomatis.
    """)



# FUNGSI-FUNGSI UTAMA 
def muat_dan_gabungkan_data(files):
    if len(files) != 3:
        st.warning("⚠️ Harap unggah tepat 3 file CSV.")
        return None

    list_df_panjang = []

    daftar_bulan = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    try:
        for file in files:
            # tahun dari nama file
            match = re.search(r'(\d{4})', file.name)
            if not match:
                st.error(f"❌ Error: Tidak dapat menemukan tahun dalam nama file '{file.name}'.")
                return None
            tahun = int(match.group(1))

            # Baca semua data
            df_all = pd.read_csv(file, header=None)

            # Ambil nama bulan dari baris ke-3 (index 3) dan isi manual kolom pertama 'Kebangsaan'
            kolom_bulan = df_all.iloc[3].tolist()
            kolom_bulan[0] = 'Kebangsaan' 

            # Data asli dimulai dari baris ke-4 (index 4)
            df_data = df_all.iloc[4:].copy()
            df_data.columns = kolom_bulan  # Tetapkan nama kolom

            # Hapus kolom 'Tahunan' jika ada
            if 'Tahunan' in df_data.columns:
                df_data.drop(columns=['Tahunan'], inplace=True)

            # Validasi kolom
            if not all(bulan in df_data.columns for bulan in daftar_bulan):
                st.error(f"❌ Error: Kolom bulan tidak lengkap dalam file '{file.name}'.")
                return None

            # Transformasi ke bentuk panjang
            df_panjang = df_data.melt(
                id_vars=['Kebangsaan'],
                value_vars=daftar_bulan,
                var_name='bulan',
                value_name=KOLOM_PENGUNJUNG
            )
            df_panjang['tahun'] = tahun
            list_df_panjang.append(df_panjang)

        # Gabung semua tahun
        df_gabungan = pd.concat(list_df_panjang, ignore_index=True)
        df_gabungan.rename(columns={'Kebangsaan': KOLOM_NEGARA}, inplace=True)

        # Filter negara yang dibutuhkan
        negara_filter = ['malaysia', 'philippines', 'singapore']
        df_filter = df_gabungan[df_gabungan[KOLOM_NEGARA].str.strip().str.lower().isin(negara_filter)].copy()

        # Bersihkan angka
        df_filter[KOLOM_PENGUNJUNG] = pd.to_numeric(df_filter[KOLOM_PENGUNJUNG], errors='coerce')
        df_filter.dropna(subset=[KOLOM_PENGUNJUNG], inplace=True)

        if df_filter.empty:
            st.error("❌ Error: Data untuk Malaysia, Philippines, atau Singapore tidak ditemukan.")
            return None

        return df_filter

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
        return None

# --- Fungsi-fungsi lain tidak perlu diubah sama sekali ---
def buat_tabel_monte_carlo(df, nama_negara_lower):
    """
    Fungsi untuk membuat tabel persiapan simulasi Monte Carlo berdasarkan data historis.
    """
    data_negara = df[df[KOLOM_NEGARA].str.lower() == nama_negara_lower][KOLOM_PENGUNJUNG]
    
    if data_negara.empty or len(data_negara) <= 1:
        st.warning(f"Data tidak cukup untuk negara {nama_negara_lower.capitalize()} untuk membuat tabel Monte Carlo.")
        return None

    # Hitung parameter histogram
    n = len(data_negara)
    min_val = data_negara.min()
    max_val = data_negara.max()
    rentang = max_val - min_val
    jumlah_kelas = math.ceil(1 + 3.3 * np.log10(n))
    panjang_kelas = math.ceil(rentang / jumlah_kelas)

    # Buat bins
    bins = [min_val + i * panjang_kelas for i in range(jumlah_kelas + 1)]
    # Koreksi agar batas bawah dan atas berurutan (batas bawah berikutnya = batas atas sebelumnya + 1)
    for i in range(1, len(bins)-1):
        bins[i] = bins[i-1] + panjang_kelas + 1

    if bins[-1] <= max_val:
        bins[-1] = int(max_val) + 1

    # Label interval
    labels = [f"{int(bins[i])} - {int(bins[i+1]-1)}" for i in range(len(bins)-1)]
    frekuensi = pd.cut(data_negara, bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()
    
    tabel_mc = pd.DataFrame({'Frekuensi': frekuensi})
    tabel_mc.index.name = 'Interval Pengunjung'
    tabel_mc.reset_index(inplace=True)

    # Nilai tengah
    nilai_tengah = [(bins[i] + bins[i+1] - 1) / 2 for i in range(len(bins)-1)]
    tabel_mc['Nilai Tengah'] = nilai_tengah

    # Probabilitas berdasarkan nilai tengah
    total_nilai_tengah = sum(nilai_tengah)
    tabel_mc['Probabilitas'] = tabel_mc['Nilai Tengah'] / total_nilai_tengah
    tabel_mc['Probabilitas'] = tabel_mc['Probabilitas'].round(2)  # dibulatkan ke 2 digit

    # Probabilitas kumulatif
    tabel_mc['Kumulatif'] = tabel_mc['Probabilitas'].cumsum()
    tabel_mc['Kumulatif'] = tabel_mc['Kumulatif'].round(2)

    # Koreksi kumulatif terakhir agar pasti 1.00
    tabel_mc.loc[tabel_mc.index[-1], 'Kumulatif'] = 1.00
    
    # Interval angka random: 1–100
    tabel_mc['Batas Bawah'] = (tabel_mc['Kumulatif'].shift(1).fillna(0) * 100).round(0).astype(int) + 1
    tabel_mc['Batas Atas'] = (tabel_mc['Kumulatif'] * 100).round(0).astype(int)

    # Koreksi batas atas terakhir agar selalu 100
    tabel_mc.loc[tabel_mc.index[-1], 'Batas Atas'] = 100

    # Kolom interval angka random
    tabel_mc['Interval Angka Random'] = tabel_mc.apply(
        lambda row: f"{row['Batas Bawah']} - {row['Batas Atas']}", axis=1
    )

    # Kolom yang ditampilkan
    return tabel_mc[['Interval Pengunjung', 'Nilai Tengah', 'Frekuensi',
                 'Probabilitas', 'Kumulatif',
                 'Batas Bawah', 'Batas Atas', 'Interval Angka Random']]

def jalankan_simulasi(tabel_mc, jumlah_simulasi):
    """
    Fungsi untuk menjalankan simulasi berdasarkan tabel persiapan Monte Carlo.
    """
    hasil_simulasi = []
    angka_random = np.random.randint(1, 101, size=jumlah_simulasi)

    for i, rand_num in enumerate(angka_random):
        # Filter baris sesuai angka random
        baris_cocok = tabel_mc[
            (rand_num >= tabel_mc['Batas Bawah']) & (rand_num < tabel_mc['Batas Atas'])
        ]
        
        # Jika tidak ditemukan (biasanya karena rand_num = 1.0), ambil baris terakhir
        if baris_cocok.empty:
            baris_interval = tabel_mc.iloc[-1]
        else:
            baris_interval = baris_cocok.iloc[0]

        prediksi_pengunjung = baris_interval['Nilai Tengah']
        hasil_simulasi.append({
            'Bulan Ke-': i + 1,
            'Angka Random': rand_num,
            'Prediksi Jumlah Pengunjung': int(prediksi_pengunjung)
        })

    return pd.DataFrame(hasil_simulasi)


# BAGIAN 2: PROSES UTAMA DAN TAMPILAN HASIL (Tidak ada perubahan di sini)
if uploaded_files:
    df_gabungan_tidy = muat_dan_gabungkan_data(uploaded_files)

    if df_gabungan_tidy is not None:
        st.header("Tabel Gabungan Data")
        st.write("Data dari 3 file CSV telah berhasil dibaca, ditransformasi, dan digabungkan.")
        st.dataframe(df_gabungan_tidy.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.divider()
        st.header("Simulasi")
        
        jumlah_simulasi = st.slider("Pilih Jumlah Bulan untuk Simulasi:", min_value=12, max_value=60, value=24, step=1)

        tab_malay, tab_phili, tab_singapore = st.tabs(["🇲🇾 Malaysia", "🇵🇭 Philippines", "🇸🇬 Singapore"])
        
        # Nama negara disesuaikan dengan data
        negara = ['malaysia', 'philippines', 'singapore']
        tabs = [tab_malay, tab_phili, tab_singapore]
        
        tabel_mc_dict = {}
        tabel_simulasi_dict = {}

        for n, tab in zip(negara, tabs):
            with tab:
                st.subheader(f"Analisis untuk {n.capitalize()}")
                
                tabel_mc = buat_tabel_monte_carlo(df_gabungan_tidy, n)
                if tabel_mc is not None:
                    tabel_mc_dict[n] = tabel_mc
                    st.write("**Tabel Persiapan Monte Carlo**")
                    st.dataframe(tabel_mc.drop(columns=['Batas Bawah', 'Batas Atas']).reset_index(drop=True), use_container_width=True, hide_index=True)

                    st.write(f"**Tabel Simulasi untuk {jumlah_simulasi} Bulan**")
                    tabel_simulasi = jalankan_simulasi(tabel_mc, jumlah_simulasi)
                    tabel_simulasi_dict[n] = tabel_simulasi
                    st.dataframe(tabel_simulasi.reset_index(drop=True), use_container_width=True, hide_index=True)

                    st.write("**Diagram Hasil Simulasi**")
                    fig = px.line(
                        tabel_simulasi, x='Bulan Ke-', y='Prediksi Jumlah Pengunjung',
                        title=f'Simulasi Kunjungan dari {n.capitalize()}', markers=True
                    )
                    fig.update_layout(xaxis_title="Bulan Simulasi", yaxis_title="Jumlah Pengunjung (Prediksi)")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.header("Analisis Variabel Turunan")

        if len(tabel_simulasi_dict) == 3:
            with st.container(border=True):
                st.subheader("Total Gabungan Kunjungan dari 3 Negara (Hasil Simulasi)")
                
                df_total_kunjungan = pd.DataFrame({'Bulan Ke-': tabel_simulasi_dict['malaysia']['Bulan Ke-']})
                df_total_kunjungan['Total Pengunjung'] = (
                    tabel_simulasi_dict['malaysia']['Prediksi Jumlah Pengunjung'] +
                    tabel_simulasi_dict['philippines']['Prediksi Jumlah Pengunjung'] +
                    tabel_simulasi_dict['singapore']['Prediksi Jumlah Pengunjung']
                )

                st.dataframe(df_total_kunjungan.reset_index(drop=True), use_container_width=True, hide_index=True)
                fig_total = px.bar(
                    df_total_kunjungan, x='Bulan Ke-', y='Total Pengunjung',
                    title='Total Gabungan Kunjungan Wisatawan per Bulan'
                )
                fig_total.update_layout(xaxis_title="Bulan Simulasi", yaxis_title="Total Jumlah Pengunjung")
                st.plotly_chart(fig_total, use_container_width=True)

            with st.container(border=True):
                st.subheader("Rata-rata Lama Tinggal per Wisatawan (Berdasarkan Asumsi)")
                st.write("Masukkan asumsi rata-rata lama tinggal (dalam hari) untuk menghitung rata-rata tertimbang.")

                #batas asumsi
                col1, col2, col3 = st.columns(3)
                asumsi_malay = col1.number_input("Lama Tinggal Turis Malaysia (hari)", min_value=1.0, value=2.0, step=0.1)
                asumsi_phili = col2.number_input("Lama Tinggal Turis Filipina (hari)", min_value=1.0, value=5.0, step=0.1)
                asumsi_singapore = col3.number_input("Lama Tinggal Turis Singapura (hari)", min_value=1.0, value=3.0, step=0.1)
                
                total_hari_tinggal = (
                    tabel_simulasi_dict['malaysia']['Prediksi Jumlah Pengunjung'] * asumsi_malay +
                    tabel_simulasi_dict['philippines']['Prediksi Jumlah Pengunjung'] * asumsi_phili +
                    tabel_simulasi_dict['singapore']['Prediksi Jumlah Pengunjung'] * asumsi_singapore
                )
                
                df_rata2_tinggal = pd.DataFrame({
                    'Bulan Ke-': df_total_kunjungan['Bulan Ke-'],
                    'Total Pengunjung': df_total_kunjungan['Total Pengunjung'],
                    'Total Hari Tinggal (Visitor-Days)': total_hari_tinggal
                })
                df_rata2_tinggal['Rata-rata Lama Tinggal (Hari)'] = df_rata2_tinggal.apply(
                    lambda row: row['Total Hari Tinggal (Visitor-Days)'] / row['Total Pengunjung'] if row['Total Pengunjung'] > 0 else 0,
                    axis=1
                ).round(2)
                
                st.write("**Tabel Rata-rata Lama Tinggal**")
                st.dataframe(df_rata2_tinggal[['Bulan Ke-', 'Rata-rata Lama Tinggal (Hari)']].reset_index(drop=True), use_container_width=True, hide_index=True)
                fig_tinggal = px.line(
                    df_rata2_tinggal, x='Bulan Ke-', y='Rata-rata Lama Tinggal (Hari)',
                    title='Rata-rata Lama Tinggal Wisatawan per Bulan (Tertimbang)', markers=True
                )
                fig_tinggal.update_layout(xaxis_title="Bulan Simulasi", yaxis_title="Rata-rata Lama Tinggal (Hari)")
                st.plotly_chart(fig_tinggal, use_container_width=True)
                
        ##wawasan tambahan
        st.divider()
        st.header("Analisis Wawasan Strategis")
        with st.container(border=True):
            st.subheader("Perbandingan Kunjungan antar Negara")

            # Hitung total pengunjung dari hasil simulasi untuk tiap negara
            total_malay = tabel_simulasi_dict['malaysia']['Prediksi Jumlah Pengunjung'].sum()
            total_phili = tabel_simulasi_dict['philippines']['Prediksi Jumlah Pengunjung'].sum()
            total_singapore = tabel_simulasi_dict['singapore']['Prediksi Jumlah Pengunjung'].sum()
            total_all = total_malay + total_phili + total_singapore

            # Hitung persentase kontribusi
            prop_malay = round((total_malay / total_all) * 100, 1)
            prop_phili = round((total_phili / total_all) * 100, 1)
            prop_singapore = round((total_singapore / total_all) * 100, 1)

            df_persen = pd.DataFrame({
                'Negara': ['Malaysia', 'Philippines', 'Singapore'],
                'Total Kunjungan': [total_malay, total_phili, total_singapore],
                'Persentase (%)': [prop_malay, prop_phili, prop_singapore]
            })

            st.dataframe(df_persen, hide_index=True, use_container_width=True)

            st.write(f"Malaysia menyumbang {prop_malay}% dari total kunjungan, sedangkan Philippines dan Singapore masing-masing menyumbang {prop_phili}% dan {prop_singapore}%. Ketimpangan ini menunjukkan potensi pasar yang belum tergarap dari negara-negara dengan proporsi lebih kecil.")

            fig_pie = px.pie(df_persen, names='Negara', values='Total Kunjungan', title='Distribusi Total Kunjungan Wisatawan per Negara')
            st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("👋 Selamat datang! Silakan mulai dengan mengunggah 3 file CSV Anda melalui panel di sebelah kiri.")