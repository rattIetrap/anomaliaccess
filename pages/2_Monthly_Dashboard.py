# pages/2_Monthly_Dashboard.py
import streamlit as st
import pandas as pd
import os
import io
import glob # Pustaka baru untuk mencari file

# --- Konfigurasi dan Fungsi ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

@st.cache_data(ttl=300) # Cache data selama 5 menit
def load_all_history_files(data_folder_path):
    """Mencari, membaca, dan menggabungkan semua file ringkasan harian."""
    if not os.path.exists(data_folder_path):
        st.error(f"Folder 'data' tidak ditemukan di path: {data_folder_path}")
        return pd.DataFrame()

    # Cari semua file .csv di dalam folder yang namanya diawali dengan 'summary_'
    all_summary_files = glob.glob(os.path.join(data_folder_path, "summary_*.csv"))
    
    if not all_summary_files:
        return None # Kembalikan None jika tidak ada file sama sekali

    df_list = []
    for f in all_summary_files:
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            df_list.append(df)
        except Exception as e:
            st.warning(f"Gagal memuat file {os.path.basename(f)}: {e}")
            continue # Lanjutkan ke file berikutnya jika ada error
            
    if not df_list:
        return pd.DataFrame()

    # Gabungkan semua DataFrame menjadi satu
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Penanganan duplikat: jika ada beberapa entri untuk tanggal yang sama,
    # ambil yang terakhir saja berdasarkan asumsi itu yang terbaru.
    combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    # Urutkan berdasarkan tanggal
    combined_df.sort_values(by='date', inplace=True)
    
    return combined_df

# ... (Fungsi convert_df_to_excel tetap sama) ...
@st.cache_data 
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_History')
    return output.getvalue()


def display_monthly_dashboard():
    st.title("📅 Dashboard Tren Deteksi Anomali")
    st.markdown("Visualisasi ini menampilkan tren deteksi anomali berdasarkan semua *file* ringkasan harian yang tersimpan.")

    if st.button("🔄 Muat Ulang Data Histori"):
        st.cache_data.clear()
        st.rerun()

    df_history = load_all_history_files(DATA_FOLDER)
    
    if df_history is None or df_history.empty:
        st.warning(
            "Tidak ada file ringkasan (`summary_*.csv`) yang ditemukan di folder 'data'. "
            "Silakan unggah file ringkasan harian ke folder 'data' di repositori GitHub Anda.", 
            icon="⚠️"
        )
        return

    # Sisa kode untuk menampilkan metrik, grafik, dan tabel tetap sama persis
    # karena ia bekerja pada DataFrame df_history yang sudah digabungkan.
    # ... (kode st.subheader("Ringkasan Periode Total"), st.metric, st.line_chart, st.dataframe, st.download_button) ...
    df_display_and_download = df_history.copy()
    df_history.set_index('date', inplace=True)
    
    st.markdown("---")
    
    st.subheader("Ringkasan Periode Total")
    total_anomalies_ae = df_history['anomaly_count_ae'].sum()
    total_anomalies_ocsvm = df_history['anomaly_count_ocsvm'].sum()
    total_logs_in_period = df_history['total_logs'].sum()
    day_with_highest_anomalies = df_history['anomaly_count_ae'].idxmax() if not df_history.empty else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Anomali (AE)", f"{total_anomalies_ae:,}")
    col2.metric("Total Anomali (OC-SVM)", f"{total_anomalies_ocsvm:,}")
    col3.metric("Total Log Diproses", f"{total_logs_in_period:,}")
    
    if pd.notna(day_with_highest_anomalies):
        st.info(f"Hari dengan anomali terbanyak (AE): **{day_with_highest_anomalies.strftime('%d %B %Y')}** "
                f"({df_history.loc[day_with_highest_anomalies, 'anomaly_count_ae']} anomali).", icon="🔥")

    st.markdown("---")
    st.subheader("Grafik Tren Anomali Harian")
    columns_to_plot = st.multiselect(
        "Pilih data untuk ditampilkan di grafik:",
        options=['Anomali (AE)', 'Anomali (OC-SVM)', 'Total Log'],
        default=['Anomali (AE)', 'Anomali (OC-SVM)']
    )
    
    plot_data = pd.DataFrame()
    if 'Anomali (AE)' in columns_to_plot: plot_data['Anomali (AE)'] = df_history['anomaly_count_ae']
    if 'Anomali (OC-SVM)' in columns_to_plot: plot_data['Anomali (OC-SVM)'] = df_history['anomaly_count_ocsvm']
    if 'Total Log' in columns_to_plot: plot_data['Total Log'] = df_history['total_logs']
        
    if not plot_data.empty:
        st.line_chart(plot_data)
        
    st.markdown("---")
    st.subheader("Data Histori Deteksi Harian (Digabungkan)")
    st.dataframe(df_display_and_download)
    
    excel_data = convert_df_to_excel(df_display_and_download)
    st.download_button(
        label="📥 Unduh Histori Gabungan (Excel)",
        data=excel_data,
        file_name="combined_detection_history.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Pastikan hanya dijalankan jika user sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Silakan login melalui halaman utama untuk mengakses dashboard ini.")
    st.stop()
else:
    display_monthly_dashboard()
