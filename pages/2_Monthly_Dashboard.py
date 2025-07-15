# pages/2_Monthly_Dashboard.py
import streamlit as st
import pandas as pd
import os
import io

# --- Konfigurasi dan Fungsi ---
# Path ke file histori master di dalam repositori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_HISTORY_PATH = os.path.join(BASE_DIR, 'data', 'master_history.csv')

@st.cache_data 
def load_master_history(path):
    """Memuat data histori master dari file CSV."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error saat memuat file histori master: {e}")
        return None

@st.cache_data 
def convert_df_to_excel(df):
    """Mengubah DataFrame menjadi file biner Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_History')
    processed_data = output.getvalue()
    return processed_data

def display_monthly_dashboard():
    st.title("📅 Dashboard Tren Deteksi Anomali")
    st.markdown("Visualisasi ini menampilkan tren deteksi anomali dari histori yang telah dikumpulkan secara manual.")

    df_history = load_master_history(MASTER_HISTORY_PATH)
    
    if df_history is None or df_history.empty:
        st.warning(
            "File histori master (`data/master_history.csv`) tidak ditemukan atau kosong. "
            "Pastikan file ini sudah dibuat dan diunggah ke repositori GitHub Anda.", 
            icon="⚠️"
        )
        return

    # Urutkan data berdasarkan tanggal untuk memastikan grafik benar
    df_history.sort_values(by='date', inplace=True)
    df_display_and_download = df_history.copy()
    df_history.set_index('date', inplace=True)
    
    st.markdown("---")
    
    # --- Tampilkan Metrik Utama ---
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
    if 'Anomali (AE)' in columns_to_plot:
        plot_data['Anomali (AE)'] = df_history['anomaly_count_ae']
    if 'Anomali (OC-SVM)' in columns_to_plot:
        plot_data['Anomali (OC-SVM)'] = df_history['anomaly_count_ocsvm']
    if 'Total Log' in columns_to_plot:
        plot_data['Total Log'] = df_history['total_logs']
        
    if not plot_data.empty:
        st.line_chart(plot_data)
        
    st.markdown("---")
    st.subheader("Data Histori Deteksi Harian")
    st.dataframe(df_display_and_download)
    
    excel_data = convert_df_to_excel(df_display_and_download)
    st.download_button(
        label="📥 Unduh Histori Ini (Excel)",
        data=excel_data,
        file_name="master_detection_history.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Pastikan hanya dijalankan jika user sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Silakan login melalui halaman utama untuk mengakses dashboard ini.")
    st.stop()
else:
    display_monthly_dashboard()
