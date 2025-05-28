# pages/1_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# Impor fungsi yang relevan dari models.py
# Kita akan menggunakan parse_log_file, get_autoencoder_anomalies, dan get_ocsvm_anomalies dari models.py
# Untuk pra-pemrosesan data prediksi, kita akan buat fungsi khusus di sini
# yang sesuai dengan logika train_script.ipynb
try:
    from models import (
        parse_log_file,
        get_autoencoder_anomalies,
        get_ocsvm_anomalies
    )
except ImportError:
    st.error("Error: File 'models.py' tidak ditemukan. Pastikan file tersebut ada di direktori root proyek.")
    st.stop()

# --- Konfigurasi Path Artefak ---
# Menyesuaikan path jika 1_Dashboard.py ada di dalam subfolder 'pages'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')

AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl")
# training_mse_ae.npy tidak disimpan secara default oleh train_script.ipynb
# Jika Anda telah memodifikasi notebook untuk menyimpannya, path ini akan digunakan.
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy")

# --- Fungsi Pra-pemrosesan Data untuk Dashboard (sesuai train_script.ipynb) ---
def preprocess_dashboard_data(df_raw, label_encoders_loaded, model_cols_trained, scaler_loaded):
    """
    Melakukan pra-pemrosesan pada DataFrame log baru untuk prediksi,
    sesuai dengan langkah-langkah di train_script.ipynb.
    """
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Kolom yang digunakan saat training (dari model_columns.pkl)
    features_to_use = model_cols_trained

    df_processed_display = df_raw.copy() # Untuk menampilkan data asli yang relevan
    df_model_input = pd.DataFrame()

    # Pastikan semua fitur yang dibutuhkan ada, isi dengan 'Unknown' jika tidak
    missing_cols_for_display = []
    for col in features_to_use:
        if col not in df_processed_display.columns:
            df_processed_display[col] = 'Unknown' # Default untuk kolom yang hilang di data input
            missing_cols_for_display.append(col)
    if missing_cols_for_display:
        st.warning(f"Kolom berikut tidak ditemukan di file log input dan diisi dengan 'Unknown': {', '.join(missing_cols_for_display)}")

    # Ambil hanya kolom yang relevan untuk model
    df_model_input = df_processed_display[features_to_use].copy()
    
    # Pra-pemrosesan seperti di train_script.ipynb (cell 5)
    # 1. Pastikan semua fitur diperlakukan sebagai string & tangani NaN
    for col in features_to_use:
        df_model_input[col] = df_model_input[col].astype(str).fillna('Unknown')

    # 2. Terapkan Label Encoding yang sudah di-load
    for col in features_to_use:
        if col in label_encoders_loaded:
            le = label_encoders_loaded[col]
            # Tangani label baru yang tidak ada saat training
            df_model_input[col] = df_model_input[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1 # -1 untuk unknown/unseen
            )
            # Jika ada -1 (unseen label), ganti dengan nilai modus atau nilai default yang konsisten
            # Di sini kita contohkan dengan mengganti -1 menjadi 0 (atau bisa nilai lain)
            # Idealnya, modus dari data training disimpan dan digunakan di sini.
            if -1 in df_model_input[col].unique():
                # st.info(f"Kolom '{col}' memiliki nilai baru yang tidak terlihat saat training. Ditandai sementara dan akan diisi dengan nilai default (misal: 0 atau modus dari training).")
                # Cari kelas 'Unknown' jika ada di encoder, jika tidak gunakan nilai default (misal 0)
                try:
                    unknown_encoded_val = le.transform(['Unknown'])[0]
                    df_model_input[col] = df_model_input[col].replace(-1, unknown_encoded_val)
                except ValueError: # Jika 'Unknown' sendiri tidak ada di .classes_
                    df_model_input[col] = df_model_input[col].replace(-1, 0) # Default fallback
        else:
            st.error(f"Label Encoder untuk kolom '{col}' tidak ditemukan. Kolom ini mungkin tidak akan diproses dengan benar.")
            df_model_input[col] = 0 # Default jika encoder tidak ada

    # 3. Terapkan Scaler yang sudah di-load
    try:
        if df_model_input.shape[1] != scaler_loaded.n_features_in_:
            st.error(f"Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan yang diharapkan scaler ({scaler_loaded.n_features_in_}). Model Columns: {features_to_use}")
            return pd.DataFrame(), df_processed_display[features_to_use]
        
        scaled_data = scaler_loaded.transform(df_model_input)
        df_scaled = pd.DataFrame(scaled_data, columns=features_to_use, index=df_model_input.index)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        return pd.DataFrame(), df_processed_display[features_to_use]
        
    return df_scaled, df_processed_display[features_to_use]


# --- Fungsi Pemuatan Artefak ---
@st.cache_resource # Cache resource agar tidak load ulang terus menerus
def load_all_artifacts():
    artifacts = {
        "autoencoder_model": None, "ocsvm_model": None, "scaler": None,
        "label_encoders": None, "model_columns": None, "training_mse_ae": None,
        "all_loaded_successfully": True
    }
    error_messages = []

    # Path yang akan ditampilkan di sidebar (tanpa BASE_MODEL_DIR)
    def display_path(full_path):
        return os.path.join(os.path.basename(MODEL_ARTIFACTS_FOLDER), os.path.basename(full_path))

    try:
        artifacts["autoencoder_model"] = load_model(AUTOENCODER_MODEL_PATH)
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat Model Autoencoder: {display_path(AUTOENCODER_MODEL_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False
    try:
        artifacts["ocsvm_model"] = joblib.load(OCSVM_MODEL_PATH)
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat Model OC-SVM: {display_path(OCSVM_MODEL_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False
    try:
        artifacts["scaler"] = joblib.load(SCALER_PATH)
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat File Scaler: {display_path(SCALER_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False
    try:
        artifacts["label_encoders"] = joblib.load(LABEL_ENCODERS_PATH)
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat File Label Encoders: {display_path(LABEL_ENCODERS_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False
    try:
        artifacts["model_columns"] = joblib.load(MODEL_COLUMNS_PATH)
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat File Kolom Model: {display_path(MODEL_COLUMNS_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False
    try:
        artifacts["training_mse_ae"] = np.load(TRAINING_MSE_AE_PATH)
    except FileNotFoundError:
        error_messages.append(f"⚠️ File training MSE Autoencoder tidak ditemukan: {display_path(TRAINING_MSE_AE_PATH)}. Threshold AE akan dihitung dari data input.")
        # Tidak set all_loaded_successfully ke False karena ini opsional
    except Exception as e:
        error_messages.append(f"🚨 Gagal memuat Training MSE AE: {display_path(TRAINING_MSE_AE_PATH)} ({e})")
        artifacts["all_loaded_successfully"] = False # Jika ada error lain selain FileNotFoundError

    # Tampilkan status di sidebar
    st.sidebar.subheader("ℹ️ Status Pemuatan Model & Artefak")
    if not error_messages and artifacts["all_loaded_successfully"]:
        st.sidebar.success("✅ Semua model dan artefak berhasil dimuat.")
        st.sidebar.caption(f"Folder Artefak: {os.path.basename(MODEL_ARTIFACTS_FOLDER)}")
        st.sidebar.caption(f"- AE: {os.path.basename(AUTOENCODER_MODEL_PATH)}")
        st.sidebar.caption(f"- OCSVM: {os.path.basename(OCSVM_MODEL_PATH)}")
        st.sidebar.caption(f"- Scaler: {os.path.basename(SCALER_PATH)}")
        st.sidebar.caption(f"- L.Encoders: {os.path.basename(LABEL_ENCODERS_PATH)}")
        st.sidebar.caption(f"- Columns: {os.path.basename(MODEL_COLUMNS_PATH)}")
        if artifacts["training_mse_ae"] is not None:
            st.sidebar.caption(f"- MSE AE: {os.path.basename(TRAINING_MSE_AE_PATH)}")
        else:
            st.sidebar.caption(f"- MSE AE: Tidak ditemukan, menggunakan fallback.")

    else:
        for msg in error_messages:
            st.sidebar.error(msg)
        if not artifacts["all_loaded_successfully"]:
             st.sidebar.error("💔 Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar.")
        st.sidebar.info(f"Pastikan semua artefak ada di folder '{os.path.basename(MODEL_ARTIFACTS_FOLDER)}' dan skrip 'train_script.ipynb' sudah dijalankan dengan sukses untuk menghasilkan file-file tersebut.")

    return artifacts

# --- Halaman Dashboard ---
def display_dashboard():
    st.title("🛡️ Dashboard Deteksi Anomali Akses Fortigate")
    st.markdown("Unggah file log Fortigate (.txt) untuk dideteksi.")

    # Panggil fungsi load_artifacts di awal
    # Jika gagal memuat artefak penting, dashboard mungkin tidak berfungsi penuh
    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = load_all_artifacts()
    
    artifacts = st.session_state.artifacts

    # Periksa apakah model penting berhasil dimuat
    critical_artifacts_loaded = (
        artifacts["autoencoder_model"] is not None and
        artifacts["ocsvm_model"] is not None and
        artifacts["scaler"] is not None and
        artifacts["label_encoders"] is not None and
        artifacts["model_columns"] is not None
    )

    if not critical_artifacts_loaded:
        st.error("Model atau artefak penting tidak berhasil dimuat. Silakan periksa status di sidebar dan pastikan file-file yang diperlukan tersedia.")
        st.warning("Anda mungkin perlu menjalankan ulang `train_script.ipynb` untuk menghasilkan artefak yang benar.")
        # Tombol untuk mencoba memuat ulang artefak
        if st.button("🔄 Coba Muat Ulang Artefak"):
            del st.session_state.artifacts # Hapus cache
            st.rerun()
        return # Hentikan eksekusi jika artefak penting tidak ada

    uploaded_file = st.file_uploader("Pilih file log (.txt)", type="txt")

    # Tombol Deteksi di luar kondisi uploaded_file agar bisa di-disable
    col1, col2 = st.columns([1,3])
    with col1:
        detect_button = st.button("🔍 Deteksi Anomali", use_container_width=True, disabled=not uploaded_file)
    
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()


    if detect_button and uploaded_file is not None:
        st.session_state.detection_done = False # Reset status
        st.session_state.results_df = pd.DataFrame()

        with st.spinner("Memproses file log dan mendeteksi anomali..."):
            # Simpan file yang diunggah sementara untuk diproses oleh parse_log_file
            temp_log_path = os.path.join(MODEL_ARTIFACTS_FOLDER, f"temp_{uploaded_file.name}")
            with open(temp_log_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            df_new_raw = parse_log_file(temp_log_path) #

            if df_new_raw.empty:
                st.error("Tidak ada data yang berhasil diparsing dari file log yang diunggah.")
                if os.path.exists(temp_log_path):
                    os.remove(temp_log_path)
                return

            st.info(f"Jumlah record log yang diparsing: {len(df_new_raw)}")

            # Pra-pemrosesan data baru menggunakan artefak yang dilatih oleh train_script.ipynb
            df_new_scaled, df_new_display = preprocess_dashboard_data(
                df_new_raw.copy(),
                artifacts["label_encoders"],
                artifacts["model_columns"], # Menggunakan kolom dari training
                artifacts["scaler"]
            )

            if df_new_scaled.empty:
                st.error("Pra-pemrosesan data baru gagal. Pastikan format log sesuai.")
                if os.path.exists(temp_log_path):
                    os.remove(temp_log_path)
                return
            
            st.success("Pra-pemrosesan data baru selesai.")

            # Deteksi Anomali dengan Autoencoder
            ae_anomalies_series, ae_mse_series = get_autoencoder_anomalies(
                artifacts["autoencoder_model"],
                df_new_scaled,
                training_mse=artifacts["training_mse_ae"] # Bisa None jika file tidak ada
            ) #

            # Deteksi Anomali dengan One-Class SVM
            ocsvm_anomalies_series, ocsvm_scores_series = get_ocsvm_anomalies(
                artifacts["ocsvm_model"],
                df_new_scaled
            ) #

            # Gabungkan hasil
            results_df = df_new_display.copy() # df_new_display berisi kolom asli yang dipilih
            results_df.reset_index(drop=True, inplace=True) # Reset index untuk konkatenasi
            
            # Pastikan Series memiliki index yang sama sebelum assignment
            ae_anomalies_series.index = results_df.index
            ae_mse_series.index = results_df.index
            ocsvm_anomalies_series.index = results_df.index
            ocsvm_scores_series.index = results_df.index

            results_df['AE_Anomaly'] = ae_anomalies_series
            results_df['AE_MSE'] = ae_mse_series
            results_df['OCSVM_Anomaly'] = ocsvm_anomalies_series
            results_df['OCSVM_Score'] = ocsvm_scores_series

            # Tentukan anomali gabungan (misalnya, jika salah satu model mendeteksinya sebagai anomali)
            results_df['Combined_Anomaly'] = results_df['AE_Anomaly'] | results_df['OCSVM_Anomaly']
            
            st.session_state.results_df = results_df
            st.session_state.detection_done = True

            # Hapus file log sementara
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)

    if st.session_state.detection_done and not st.session_state.results_df.empty:
        results_df_display = st.session_state.results_df
        st.subheader("📈 Hasil Deteksi Anomali")

        # Tampilkan ringkasan
        total_records = len(results_df_display)
        ae_anomalies_count = results_df_display['AE_Anomaly'].sum()
        ocsvm_anomalies_count = results_df_display['OCSVM_Anomaly'].sum()
        combined_anomalies_count = results_df_display['Combined_Anomaly'].sum()

        st.metric(label="Total Records Diproses", value=total_records)
        st.metric(label="Anomali Terdeteksi oleh Autoencoder", value=ae_anomalies_count)
        st.metric(label="Anomali Terdeteksi oleh OC-SVM", value=ocsvm_anomalies_count)
        st.metric(label="Anomali Gabungan (AE atau OC-SVM)", value=combined_anomalies_count)

        # Tampilkan hanya data anomali gabungan
        anomalies_data_to_show = results_df_display[results_df_display['Combined_Anomaly']].copy()
        
        if not anomalies_data_to_show.empty:
            st.write("Data Anomali yang Terdeteksi:")
            # Tampilkan kolom yang relevan untuk analisis
            display_cols_anomaly = artifacts["model_columns"] + ['AE_Anomaly', 'AE_MSE', 'OCSVM_Anomaly', 'OCSVM_Score']
            st.dataframe(anomalies_data_to_show[display_cols_anomaly])

            # Opsi download
            @st.cache_data # Cache data untuk konversi CSV
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_anomalies = convert_df_to_csv(anomalies_data_to_show)
            st.download_button(
                label="📥 Unduh Data Anomali (CSV)",
                data=csv_anomalies,
                file_name=f"anomalies_detected_{uploaded_file.name if uploaded_file else 'report'}.csv",
                mime="text/csv",
            )
        else:
            st.success("🎉 Tidak ada anomali yang terdeteksi berdasarkan kriteria gabungan.")

        # Tombol untuk menampilkan semua data hasil (termasuk yang bukan anomali)
        if st.checkbox("Tampilkan Semua Data Hasil (Termasuk Normal)"):
            display_cols_all = artifacts["model_columns"] + ['AE_Anomaly', 'AE_MSE', 'OCSVM_Anomaly', 'OCSVM_Score', 'Combined_Anomaly']
            st.dataframe(results_df_display[display_cols_all])
            csv_all = convert_df_to_csv(results_df_display)
            st.download_button(
                label="📥 Unduh Semua Data Hasil (CSV)",
                data=csv_all,
                file_name=f"all_detection_results_{uploaded_file.name if uploaded_file else 'report'}.csv",
                mime="text/csv",
            )
    elif detect_button and uploaded_file is not None and st.session_state.results_df.empty and not st.session_state.detection_done:
        # Kasus di mana proses selesai tapi tidak ada hasil (misalnya error saat proses)
        st.info("Proses deteksi selesai, namun tidak ada hasil untuk ditampilkan. Periksa pesan error di atas.")


# --- Kontrol Utama ---
if __name__ == "__main__":
    # Cek status login dari session_state yang diatur oleh streamlit_app.py
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("🔒 Anda harus login untuk mengakses halaman ini.")
        st.info("Silakan kembali ke halaman utama untuk login.")
        # Tambahkan link atau tombol kembali ke halaman login jika memungkinkan
        # st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
    else:
        st.sidebar.success(f"Login sebagai: {st.session_state.get('username', 'Pengguna')}")
        st.sidebar.markdown("---")
        
        # Tombol logout di sidebar
        if st.sidebar.button("Logout", key="dashboard_logout_button_sidebar", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["dashboard_loaded_once"] = False # Dari streamlit_app.py
            if "login_error" in st.session_state: 
                del st.session_state["login_error"]
            # Hapus state spesifik dashboard saat logout
            if "results_df" in st.session_state:
                del st.session_state["results_df"]
            if "detection_done" in st.session_state:
                del st.session_state.detection_done
            if "artifacts" in st.session_state: # Hapus cache artefak saat logout
                del st.session_state.artifacts
            st.rerun() # Untuk kembali ke halaman login (jika halaman utama adalah login)

        display_dashboard()
