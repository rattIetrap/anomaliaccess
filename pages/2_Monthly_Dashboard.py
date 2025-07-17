# pages/2_Monthly_Dashboard.py
import streamlit as st
import pandas as pd
import os
import io
import glob
import calendar

# --- Konfigurasi dan Fungsi ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

@st.cache_data(ttl=300) # Cache data selama 5 menit
def get_available_periods(data_folder_path):
    """
    Mencari semua file ringkasan dan mengekstrak tahun serta bulan yang tersedia.
    """
    if not os.path.exists(data_folder_path):
        return {}
    
    # Pola untuk mencari file summary harian
    summary_files = glob.glob(os.path.join(data_folder_path, "summary_*.csv"))
    
    periods = {}
    for f in summary_files:
        try:
            # Ekstrak tanggal dari nama file, misal: 'summary_2024-10-15.csv'
            date_str = os.path.basename(f).replace('summary_', '').replace('.csv', '')
            # Ambil hanya tahun dan bulan
            year, month = int(date_str.split('-')[0]), int(date_str.split('-')[1])
            
            if year not in periods:
                periods[year] = []
            if month not in periods[year]:
                periods[year].append(month)
        except (ValueError, IndexError):
            continue # Abaikan file dengan format nama yang salah
            
    # Urutkan bulan untuk setiap tahun (dari terbaru ke terlama)
    for year in periods:
        periods[year].sort(reverse=True)
        
    return periods

@st.cache_data(ttl=300)
def load_and_combine_history(data_folder_path, year, month):
    """
    Memuat dan menggabungkan semua file ringkasan harian untuk tahun dan bulan tertentu.
    """
    month_str = f"{month:02d}" # Format bulan menjadi dua digit (misal: 9 -> '09')
    file_pattern = os.path.join(data_folder_path, f"summary_{year}-{month_str}-*.csv")
    
    all_files_for_period = glob.glob(file_pattern)
    
    if not all_files_for_period:
        return pd.DataFrame()

    df_list = []
    for f in all_files_for_period:
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            df_list.append(df)
        except Exception as e:
            st.warning(f"Gagal memuat file {os.path.basename(f)}: {e}")
            continue
    
    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    combined_df.sort_values(by='date', inplace=True)
    
    return combined_df

@st.cache_data 
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_History')
    return output.getvalue()

def display_monthly_dashboard():
    st.title("📅 Dashboard Tren Deteksi Anomali")
    st.markdown("Visualisasi ini menampilkan tren deteksi anomali berdasarkan semua file ringkasan harian yang tersimpan.")

    available_periods = get_available_periods(DATA_FOLDER)

    if not available_periods:
        st.warning(
            "Tidak ada file ringkasan (`summary_*.csv`) yang ditemukan di folder 'data'. "
            "Silakan unduh ringkasan dari 'Dashboard Deteksi Harian' dan unggah ke folder 'data' di repositori GitHub.", 
            icon="⚠️"
        )
        return

    # --- Filter Tahun dan Bulan ---
    col1, col2 = st.columns(2)
    with col1:
        available_years = sorted(available_periods.keys(), reverse=True)
        selected_year = st.selectbox("Pilih Tahun:", available_years)

    with col2:
        if selected_year:
            available_months = available_periods[selected_year]
            month_names = {month_num: calendar.month_name[month_num] for month_num in available_months}
            selected_month_num = st.selectbox(
                "Pilih Bulan:", 
                available_months,
                format_func=lambda x: month_names[x]
            )

    # Memuat data berdasarkan filter
    df_history = None
    if selected_year and selected_month_num:
        df_history = load_and_combine_history(DATA_FOLDER, selected_year, selected_month_num)

    if df_history is None or df_history.empty:
        st.info(f"Tidak ada data histori untuk periode {month_names.get(selected_month_num, '')} {selected_year}.")
        return
        
    df_display_and_download = df_history.copy()
    df_history.set_index('date', inplace=True)
    
    st.markdown("---")
    
    # --- Tampilkan Metrik, Grafik, Tabel, dan Tombol Unduh ---
    st.subheader(f"Ringkasan Periode: {month_names.get(selected_month_num, '')} {selected_year}")
    
    total_anomalies_ae = df_history['anomaly_count_ae'].sum()
    total_anomalies_ocsvm = df_history['anomaly_count_ocsvm'].sum()
    total_logs_in_period = df_history['total_logs'].sum()
    day_with_highest_anomalies = df_history['anomaly_count_ae'].idxmax() if not df_history.empty else None

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Anomali (AE)", f"{total_anomalies_ae:,}")
    metric_col2.metric("Total Anomali (OC-SVM)", f"{total_anomalies_ocsvm:,}")
    metric_col3.metric("Total Log Diproses", f"{total_logs_in_period:,}")
    
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
        label="📥 Unduh Histori Bulan Ini (Excel)",
        data=excel_data,
        file_name=f"history_summary_{selected_year}-{selected_month_num:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Pastikan hanya dijalankan jika user sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Silakan login melalui halaman utama untuk mengakses dashboard ini.")
    st.stop()
else:
    display_monthly_dashboard()
