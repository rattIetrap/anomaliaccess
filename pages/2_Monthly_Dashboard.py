import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi dan Fungsi ---
# Path ke folder data di mana file histori disimpan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

@st.cache_data # Cache data agar tidak perlu load ulang setiap interaksi
def load_history_data(path):
    """Memuat data histori deteksi harian dari file CSV."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error saat memuat file histori {os.path.basename(path)}: {e}")
        return None

def get_history_files(data_folder):
    """Mendapatkan daftar file histori yang tersedia di folder data."""
    if not os.path.exists(data_folder):
        return []
    files = [f for f in os.listdir(data_folder) if f.startswith('history_') and f.endswith('.csv')]
    return sorted(files, reverse=True) # Urutkan dari yang terbaru (misal: 2024-11, 2024-10)

def display_monthly_dashboard():
    st.title("📅 Dashboard Tren Deteksi Anomali")
    st.markdown("Visualisasi ini menampilkan tren deteksi anomali berdasarkan hasil yang disimpan dari setiap deteksi harian.")

    history_files = get_history_files(DATA_FOLDER)

    if not history_files:
        st.warning(
            "File histori deteksi tidak ditemukan di folder 'data'. "
            "Silakan lakukan deteksi pada log harian terlebih dahulu melalui **'1_Dashboard Deteksi Harian'** dan klik **'Simpan Ringkasan ke Histori'**.", 
            icon="⚠️"
        )
        return

    # --- Pilihan Periode Bulan ---
    selected_file = st.selectbox(
        "Pilih Periode (Bulan) untuk Ditampilkan:", 
        history_files,
        format_func=lambda x: x.replace('history_', '').replace('.csv', '') # Tampilkan nama yang lebih bersih
    )
    
    if selected_file:
        file_path = os.path.join(DATA_FOLDER, selected_file)
        df_history = load_history_data(file_path)

        if df_history is None or df_history.empty:
            st.error(f"Gagal memuat atau file {selected_file} kosong.")
            return
        
        # Set tanggal sebagai index untuk kemudahan plotting
        df_history.set_index('date', inplace=True)
        
        st.markdown("---")

        # --- Tampilkan Metrik Utama ---
        st.subheader(f"Ringkasan Periode: {selected_file.replace('history_', '').replace('.csv', '')}")
        total_anomalies_ae = df_history['anomaly_count_ae'].sum()
        total_anomalies_ocsvm = df_history['anomaly_count_ocsvm'].sum()
        total_logs_in_period = df_history['total_logs'].sum()
        # Ambil hari dengan anomali terbanyak berdasarkan model AE sebagai acuan utama
        day_with_highest_anomalies = df_history['anomaly_count_ae'].idxmax() if not df_history['anomaly_count_ae'].empty else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomali Terdeteksi (AE)", f"{total_anomalies_ae:,}")
        col2.metric("Total Anomali Terdeteksi (OC-SVM)", f"{total_anomalies_ocsvm:,}")
        col3.metric("Total Log Diproses", f"{total_logs_in_period:,}")
        
        if pd.notna(day_with_highest_anomalies):
            st.info(f"Hari dengan anomali terbanyak (menurut AE): **{day_with_highest_anomalies.strftime('%d %B %Y')}** "
                    f"({df_history.loc[day_with_highest_anomalies, 'anomaly_count_ae']} anomali).", icon="🔥")

        st.markdown("---")

        # --- Tampilkan Grafik ---
        st.subheader("Grafik Tren Anomali Harian")
        
        # Pilih kolom yang akan ditampilkan
        columns_to_plot = st.multiselect(
            "Pilih data untuk ditampilkan di grafik:",
            options=['Anomali (AE)', 'Anomali (OC-SVM)', 'Total Log'],
            default=['Anomali (AE)', 'Anomali (OC-SVM)']
        )
        
        plot_data = pd.DataFrame()
        if 'Anomali (AE)' in columns_to_plot:
            plot_data['Anomali (AE)'] = df_history['anomaly_count_ae']
        if 'Anomali (OC-SVM)' in columns_to_plot:
            plot_data['Anomali (OC-SVM)'] = df_history['anomaly_count_ocsvm']
        if 'Total Log' in columns_to_plot:
            plot_data['Total Log'] = df_history['total_logs']

        if not plot_data.empty:
            st.line_chart(plot_data)
            st.caption("Gunakan *mouse* untuk *zoom* dan melihat detail pada grafik.")
        else:
            st.info("Pilih setidaknya satu data untuk ditampilkan pada grafik.")
            
        st.markdown("---")
        
        # --- Tampilkan Tabel Data ---
        st.subheader("Data Histori Deteksi Harian")
        # Reset index agar tanggal kembali menjadi kolom untuk tampilan tabel
        st.dataframe(df_history.reset_index())

# Pastikan hanya dijalankan jika user sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Silakan login melalui halaman utama untuk mengakses dashboard ini.")
    st.stop()
else:
    display_monthly_dashboard()
