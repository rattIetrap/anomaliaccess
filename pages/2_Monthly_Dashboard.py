# pages/2_Monthly_Dashboard.py
import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_gsheets import GSheetsConnection

# --- Fungsi untuk Memuat Data Histori dari Google Sheets ---
@st.cache_data(ttl=300) # Cache data selama 5 menit (300 detik)
def load_history_from_gsheet():
    """Memuat data histori deteksi dari Google Sheets menggunakan st.secrets."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Ambil nama worksheet dari secrets, dengan fallback jika tidak ada
        worksheet_name = st.secrets.get("connections", {}).get("gsheets", {}).get("worksheet", "History")
        
        df = conn.read(worksheet=worksheet_name, usecols=list(range(5))) # Baca 5 kolom pertama
        
        # Hapus baris yang semua kolomnya kosong (sering terjadi di gsheets)
        df.dropna(how='all', inplace=True)
        
        if df.empty:
            return pd.DataFrame()

        # Konversi dan pastikan tipe data benar
        df['date'] = pd.to_datetime(df['date'])
        for col in ['total_logs', 'anomaly_count_ae', 'anomaly_count_ocsvm']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error saat memuat data histori dari Google Sheets: {e}")
        st.info("Pastikan konfigurasi `secrets.toml` Anda benar dan Google Sheet telah dibagikan dengan email service account.")
        return pd.DataFrame()

# --- Fungsi untuk Konversi DataFrame ke Excel ---
@st.cache_data 
def convert_df_to_excel(df):
    """Mengubah DataFrame menjadi file biner Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_History')
    processed_data = output.getvalue()
    return processed_data

# --- Halaman Dashboard Utama ---
def display_monthly_dashboard():
    st.title("📅 Dashboard Tren Deteksi Anomali")
    st.markdown("Visualisasi ini menampilkan tren deteksi anomali berdasarkan histori yang tersimpan di Google Sheets.")

    # Tombol untuk refresh data histori secara manual
    if st.button("🔄 Muat Ulang Data Histori"):
        st.cache_data.clear() # Membersihkan cache agar data terbaru diambil
        st.rerun()

    df_history = load_history_from_gsheet()
    
    if df_history is None or df_history.empty:
        st.warning(
            "Tidak ada data histori deteksi yang ditemukan di Google Sheet. "
            "Silakan lakukan deteksi pada log harian terlebih dahulu melalui **'1_Dashboard Deteksi Harian'** dan klik **'Simpan ke Histori'**.", 
            icon="⚠️"
        )
        return

    # Urutkan data berdasarkan tanggal untuk memastikan grafik benar
    df_history.sort_values(by='date', inplace=True)
    
    # Simpan DataFrame yang akan ditampilkan/diunduh sebelum set index
    df_display_and_download = df_history.copy()
    
    # Set tanggal sebagai index untuk kemudahan plotting
    df_history.set_index('date', inplace=True)
    
    st.markdown("---")

    # --- Tampilkan Metrik Utama ---
    st.subheader("Ringkasan Periode")
    total_anomalies_ae = df_history['anomaly_count_ae'].sum()
    total_anomalies_ocsvm = df_history['anomaly_count_ocsvm'].sum()
    total_logs_in_period = df_history['total_logs'].sum()
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
    st.dataframe(df_display_and_download)
    
    # --- Tombol Unduh Excel ---
    excel_data = convert_df_to_excel(df_display_and_download)
    st.download_button(
        label="📥 Unduh Histori Ini (Excel)",
        data=excel_data,
        file_name=f"history_summary_deteksi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_monthly_history_final"
    )

# Pastikan hanya dijalankan jika user sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Silakan login melalui halaman utama untuk mengakses dashboard ini.")
    st.stop()
else:
    # Inisialisasi untuk pengujian lokal jika diperlukan
    if "username" not in st.session_state:
        st.session_state.username = "Penguji Dashboard"
    display_monthly_dashboard()
