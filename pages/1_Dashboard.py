# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import time
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf # Dipertahankan jika custom_objects diperlukan, meskipun untuk .keras biasanya tidak

# Impor fungsi dari models.py
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Kita masih menggunakan parse_log_file dan fungsi deteksi anomali dari models.py
    from models import parse_log_file, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    if 'streamlit_app_run_first' not in st.session_state:
        st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root. Error: {e}")
        st.info("Jika Anda menjalankan halaman ini secara langsung (misalnya untuk debugging), coba jalankan `streamlit_app.py` terlebih dahulu untuk inisialisasi session state yang mungkin dibutuhkan.")
        st.stop()
    else:
        st.error(f"Gagal mengimpor modul 'models'. Error: {e}. Pastikan 'models.py' tidak ada error internal dan berada di direktori root proyek.")
        st.stop()

# --- Konfigurasi Path ---
BASE_DIR = project_root
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- PERUBAHAN NAMA FILE ARTEFAK ---
# Path ke Model dan Artefak (disesuaikan dengan output train_script.ipynb)
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras") # .keras
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl") # .pkl
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl") # .pkl
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl") # .pkl
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl") # Baru ditambahkan
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy") # Opsional, train_script.ipynb tidak menyimpan ini by default

# --- Fungsi Pemuatan Model dengan Cache Streamlit ---
@st.cache_resource
def load_anomaly_models_and_artifacts():
    """Memuat semua model dan artefak yang dibutuhkan."""
    models_artifacts = {"loaded_successfully": True, "messages": []}

    def check_and_load(path, name, load_func, type_name, icon, is_tf_model=False):
        if os.path.exists(path):
            try:
                if is_tf_model: # Untuk load_model dari TensorFlow
                    models_artifacts[name] = load_func(path) # custom_objects biasanya tidak diperlukan untuk .keras dengan loss standar
                else: # Untuk joblib.load atau np.load
                    models_artifacts[name] = load_func(path)
                models_artifacts["messages"].append(("success", f"{type_name} '{os.path.basename(path)}' berhasil dimuat.", icon))
            except Exception as e:
                full_error_message = f"Gagal memuat {type_name} '{os.path.basename(path)}': {str(e)}"
                print(full_error_message)
                models_artifacts["messages"].append(("error", full_error_message, "🔥"))
                models_artifacts[name] = None
                models_artifacts["loaded_successfully"] = False
        else:
            models_artifacts["messages"].append(("error", f"File {type_name} tidak ditemukan: {os.path.basename(path)}", "🚨"))
            models_artifacts[name] = None
            models_artifacts["loaded_successfully"] = False

    # --- PERUBAHAN CARA MEMUAT ---
    check_and_load(AUTOENCODER_MODEL_PATH, "autoencoder", load_model, "Model Autoencoder", "🤖", is_tf_model=True)
    check_and_load(OCSVM_MODEL_PATH, "ocsvm", joblib.load, "Model OC-SVM", "🧩")
    check_and_load(SCALER_PATH, "scaler", joblib.load, "Scaler", "⚙️")
    check_and_load(LABEL_ENCODERS_PATH, "label_encoders", joblib.load, "Label Encoders", "🏷️")
    check_and_load(MODEL_COLUMNS_PATH, "model_columns", joblib.load, "Kolom Model", "📊") # Memuat model_columns.pkl

    if os.path.exists(TRAINING_MSE_AE_PATH):
        try:
            models_artifacts["training_mse_ae"] = np.load(TRAINING_MSE_AE_PATH)
            models_artifacts["messages"].append(("success", f"Training MSE AE '{os.path.basename(TRAINING_MSE_AE_PATH)}' berhasil dimuat.", "📊"))
        except Exception as e:
            models_artifacts["messages"].append(("error", f"Gagal memuat Training MSE AE '{os.path.basename(TRAINING_MSE_AE_PATH)}': {e}", "🔥"))
            models_artifacts["training_mse_ae"] = None
    else:
        models_artifacts["messages"].append(("warning", f"File training MSE Autoencoder tidak ditemukan: {os.path.basename(TRAINING_MSE_AE_PATH)}. Threshold AE akan dihitung dari data input jika file ini tidak ada.", "⚠️"))
        models_artifacts["training_mse_ae"] = None
            
    return models_artifacts

# --- Fungsi untuk Konversi DataFrame ke CSV ---
@st.cache_data 
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# pages/1_Dashboard.py
# ... (impor dan fungsi load_anomaly_models_and_artifacts tetap sama seperti respons sebelumnya) ...
# ... (fungsi preprocess_dashboard_data juga tetap sama) ...

# --- Halaman Dashboard ---
def run_dashboard_page():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop() 

    st.title("🚀 Dashboard Perbandingan Model Deteksi Anomali Akses Jaringan")
    
    models_artifacts = load_anomaly_models_and_artifacts() # Memanggil fungsi yang sudah disesuaikan
    
    # ... (kode untuk expander status pemuatan model tetap sama) ...

    critical_artifacts_missing = not (
        models_artifacts.get("autoencoder") and
        models_artifacts.get("ocsvm") and
        models_artifacts.get("scaler") and
        models_artifacts.get("label_encoders") and
        models_artifacts.get("model_columns")
    )

    if critical_artifacts_missing:
        st.error("...") # Pesan error jika artefak kritis hilang
        # ... (tombol coba muat ulang) ...
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(...) # Konfigurasi uploader tetap

    if uploaded_file is not None:
        # ... (kode untuk menyimpan file temporer tetap sama) ...

        st.markdown("---")
        st.header("2. Opsi Deteksi & Proses")
        
        # ... (kode checkbox untuk memilih model tetap sama) ...

        if st.button("Proses Log Sekarang 🔎", ...): # Tombol proses
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            else:
                with st.spinner("Sedang memproses log..."):
                    process_start_time = time.time()
                    try:
                        # Simpan DataFrame mentah asli beserta indeksnya
                        df_raw_original_with_index = parse_log_file(temp_input_filepath).reset_index() #
                        
                        if df_raw_original_with_index.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                            # ... (hapus file temp) ...
                        else:
                            df_for_preprocessing = df_raw_original_with_index.copy()
                            
                            df_scaled, _ = preprocess_dashboard_data( # Fungsi ini dari respons saya sebelumnya
                                df_for_preprocessing, # Kirim df yang sudah di-reset index
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("model_columns"),
                                models_artifacts.get("scaler")
                            )

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            else:
                                st.session_state["df_raw_original_with_index"] = df_raw_original_with_index
                                st.session_state["df_scaled_for_detection"] = df_scaled
                                st.session_state["run_ae_flag"] = run_autoencoder
                                st.session_state["run_ocsvm_flag"] = run_ocsvm
                                
                                # Deteksi anomali akan dilakukan di luar blok ini untuk menampilkan metrik dulu
                                st.success(f"Parsing dan pra-pemrosesan selesai! Waktu: {time.time() - process_start_time:.2f} detik.", icon="🎉")
                                st.session_state["detection_ready"] = True
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                    finally:
                        # ... (hapus file temp) ...
            # st.rerun() # Mungkin tidak perlu jika state diatur dengan benar


    # Bagian untuk menampilkan hasil dan metrik (setelah tombol "Proses" ditekan dan detection_ready=True)
    if st.session_state.get("detection_ready", False):
        st.markdown("---")
        st.header("3. Hasil Deteksi")

        df_raw_original = st.session_state.get("df_raw_original_with_index", pd.DataFrame())
        df_scaled = st.session_state.get("df_scaled_for_detection", pd.DataFrame())
        run_ae = st.session_state.get("run_ae_flag", False)
        run_ocsvm = st.session_state.get("run_ocsvm_flag", False)

        total_records = len(df_raw_original)
        st.metric(label="Total Records Diproses", value=total_records)

        # --- Hasil Autoencoder ---
        if run_ae and models_artifacts.get("autoencoder") and not df_scaled.empty:
            with st.spinner("Mendeteksi anomali dengan Autoencoder..."):
                ae_anomalies_series, ae_mse_series = get_autoencoder_anomalies(
                    models_artifacts["autoencoder"],
                    df_scaled,
                    training_mse=models_artifacts.get("training_mse_ae")
                ) #
            
            ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index
            ae_anomalies_count = len(ae_anomalies_indices)
            st.metric(label="Anomali Terdeteksi oleh Autoencoder", value=ae_anomalies_count)

            if ae_anomalies_count > 0:
                st.subheader(f"📜 Log Anomali - Autoencoder ({ae_anomalies_count} Log)")
                # Ambil log mentah berdasarkan indeks anomali
                # df_raw_original sudah memiliki kolom 'index' dari reset_index()
                # ae_anomalies_indices adalah indeks dari df_scaled, yang seharusnya sinkron
                anomalous_logs_ae_df = df_raw_original[df_raw_original.index.isin(ae_anomalies_indices)]
                
                # Untuk menampilkan format log mentah:
                # Kita perlu merekonstruksi string log dari dictionary di setiap baris DataFrame mentah.
                # Kolom 'level' dari df_raw_original digunakan sebagai contoh, mungkin perlu disesuaikan
                # untuk mendapatkan kolom yang berisi log mentah jika ada, atau merekonstruksinya.

                # Jika df_raw_original adalah hasil parse_log_file, tiap baris adalah dict.
                # Kita perlu cara untuk mendapatkan representasi string mentah.
                # Jika tidak ada kolom log mentah, kita tampilkan dictionary-nya
                
                for idx in anomalous_logs_ae_df.index:
                    log_dict = anomalous_logs_ae_df.loc[idx].to_dict()
                    # Hapus kolom 'index' yang ditambahkan dari reset_index jika ada
                    log_dict.pop('index', None) 
                    
                    # Rekonstruksi format key=value (ini adalah upaya, mungkin perlu penyesuaian)
                    log_str_parts = []
                    for key, value in log_dict.items():
                        if pd.notna(value) and value != '': # Hanya tampilkan field yang ada isinya
                            if isinstance(value, str) and (' ' in value or value == ''):
                                log_str_parts.append(f'{key}="{value}"')
                            else:
                                log_str_parts.append(f'{key}={value}')
                    reconstructed_log_str = " ".join(log_str_parts)
                    st.code(reconstructed_log_str, language="log")
                
                # Opsi download untuk Autoencoder (misalnya, dalam CSV dengan detail)
                df_ae_anomalies_details = df_raw_original.loc[ae_anomalies_indices].copy()
                df_ae_anomalies_details['AE_MSE'] = ae_mse_series[ae_anomalies_indices].values
                csv_ae = convert_df_to_csv(df_ae_anomalies_details)
                st.download_button(
                    label="📥 Unduh Detail Anomali AE (CSV)",
                    data=csv_ae,
                    file_name=f"anomalies_AE_{uploaded_file.name}.csv",
                    mime="text/csv",
                    key="download_ae"
                )
            else:
                st.info("Tidak ada anomali terdeteksi oleh Autoencoder.")
            st.markdown("---")

        # --- Hasil One-Class SVM ---
        if run_ocsvm and models_artifacts.get("ocsvm") and not df_scaled.empty:
            with st.spinner("Mendeteksi anomali dengan One-Class SVM..."):
                ocsvm_anomalies_series, ocsvm_scores_series = get_ocsvm_anomalies(
                    models_artifacts["ocsvm"],
                    df_scaled
                ) #

            ocsvm_anomalies_indices = ocsvm_anomalies_series[ocsvm_anomalies_series == True].index
            ocsvm_anomalies_count = len(ocsvm_anomalies_indices)
            st.metric(label="Anomali Terdeteksi oleh OC-SVM", value=ocsvm_anomalies_count)

            if ocsvm_anomalies_count > 0:
                st.subheader(f"📜 Log Anomali - OC-SVM ({ocsvm_anomalies_count} Log)")
                anomalous_logs_ocsvm_df = df_raw_original[df_raw_original.index.isin(ocsvm_anomalies_indices)]
                
                for idx in anomalous_logs_ocsvm_df.index:
                    log_dict = anomalous_logs_ocsvm_df.loc[idx].to_dict()
                    log_dict.pop('index', None)
                    log_str_parts = []
                    for key, value in log_dict.items():
                        if pd.notna(value) and value != '':
                            if isinstance(value, str) and (' ' in value or value == ''):
                                log_str_parts.append(f'{key}="{value}"')
                            else:
                                log_str_parts.append(f'{key}={value}')
                    reconstructed_log_str = " ".join(log_str_parts)
                    st.code(reconstructed_log_str, language="log")

                df_ocsvm_anomalies_details = df_raw_original.loc[ocsvm_anomalies_indices].copy()
                df_ocsvm_anomalies_details['OCSVM_Score'] = ocsvm_scores_series[ocsvm_anomalies_indices].values
                csv_ocsvm = convert_df_to_csv(df_ocsvm_anomalies_details)
                st.download_button(
                    label="📥 Unduh Detail Anomali OC-SVM (CSV)",
                    data=csv_ocsvm,
                    file_name=f"anomalies_OCSVM_{uploaded_file.name}.csv",
                    mime="text/csv",
                    key="download_ocsvm"
                )
            else:
                st.info("Tidak ada anomali terdeteksi oleh OC-SVM.")
            st.markdown("---")
        
        # Reset flag agar tidak proses ulang otomatis saat interaksi lain
        st.session_state["detection_ready"] = False 

    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")

# Panggil fungsi utama
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.username = "Penguji Dashboard"
    run_dashboard_page()
