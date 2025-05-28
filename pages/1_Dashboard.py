# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import time
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

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

# --- Fungsi Pra-pemrosesan Data untuk Dashboard (sesuai train_script.ipynb) ---
# PASTIKAN FUNGSI INI ADA DAN BENAR
# Jika belum ada, Anda perlu menambahkannya dari respons saya sebelumnya atau menyesuaikannya
# Contoh kerangka:
def preprocess_dashboard_data(df_raw, label_encoders_loaded, model_cols_trained, scaler_loaded):
    """
    Melakukan pra-pemrosesan pada DataFrame log baru untuk prediksi,
    sesuai dengan langkah-langkah di train_script.ipynb.
    """
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    features_to_use = model_cols_trained 

    df_processed_display = df_raw.copy()
    df_model_input = pd.DataFrame()

    missing_cols_for_display = []
    for col in features_to_use:
        if col not in df_processed_display.columns:
            df_processed_display[col] = 'Unknown'
            missing_cols_for_display.append(col)
    if missing_cols_for_display:
        st.warning(f"Kolom berikut tidak ditemukan di file log input dan diisi dengan 'Unknown': {', '.join(missing_cols_for_display)}")

    df_model_input = df_processed_display[features_to_use].copy()
    
    for col in features_to_use:
        df_model_input[col] = df_model_input[col].astype(str).fillna('Unknown')

    for col in features_to_use:
        if col in label_encoders_loaded:
            le = label_encoders_loaded[col]
            df_model_input[col] = df_model_input[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            if -1 in df_model_input[col].unique():
                try: 
                    unknown_val_enc = le.transform(['Unknown'])[0]
                    df_model_input[col] = df_model_input[col].replace(-1, unknown_val_enc)
                except ValueError: 
                    df_model_input[col] = df_model_input[col].replace(-1, 0) 
        else:
            st.error(f"Label Encoder untuk kolom '{col}' tidak ditemukan.")
            df_model_input[col] = 0 

    try:
        if df_model_input.shape[1] != scaler_loaded.n_features_in_:
            st.error(f"Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan yang diharapkan scaler ({scaler_loaded.n_features_in_}).")
            return pd.DataFrame(), df_processed_display[features_to_use]
        
        scaled_data_values = scaler_loaded.transform(df_model_input)
        df_scaled = pd.DataFrame(scaled_data_values, columns=features_to_use, index=df_model_input.index)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        return pd.DataFrame(), df_processed_display[features_to_use]
        
    return df_scaled, df_processed_display[features_to_use]

# --- Halaman Dashboard ---
def run_dashboard_page():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop() 

    st.title("🚀 Dashboard Perbandingan Model Deteksi Anomali Akses Jaringan")
    
    models_artifacts = load_anomaly_models_and_artifacts()
    
    with st.expander("ℹ️ Status Pemuatan Model & Artefak", expanded=not models_artifacts["loaded_successfully"]):
        for type_msg, msg, icon in models_artifacts.get("messages", []):
            if type_msg == "success": st.success(msg, icon=icon)
            elif type_msg == "error": st.error(msg, icon=icon)
            elif type_msg == "warning": st.warning(msg, icon=icon)

    critical_artifacts_missing = not (
        models_artifacts.get("autoencoder") and
        models_artifacts.get("ocsvm") and
        models_artifacts.get("scaler") and
        models_artifacts.get("label_encoders") and
        models_artifacts.get("model_columns")
    )

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar.", icon="💔")
        # Tambahkan tombol coba muat ulang jika diinginkan
        # if st.button("🔄 Coba Muat Ulang Artefak"):
        #     if "artifacts" in st.session_state:
        #         del st.session_state.artifacts
        #     st.rerun()
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", 
        type=["txt", "log"], 
        key="file_uploader_dashboard_main_unique_key_v2", # Ganti key jika ada duplikasi
        help="Unggah file log Fortigate Anda dalam format .txt atau .log untuk dianalisis."
    )

    if uploaded_file is not None:
        st.markdown(f"File yang diunggah: `{uploaded_file.name}` (`{uploaded_file.size / 1024:.2f} KB`)")
        
        unique_id = str(uuid.uuid4().hex[:8])
        temp_input_filename = f"{unique_id}_{uploaded_file.name}"
        temp_input_filepath = os.path.join(UPLOAD_FOLDER, temp_input_filename)
        
        with open(temp_input_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.header("2. Opsi Deteksi & Proses")
        
        ae_available = models_artifacts.get("autoencoder") is not None and models_artifacts.get("scaler") is not None and models_artifacts.get("label_encoders") is not None and models_artifacts.get("model_columns") is not None
        ocsvm_available = models_artifacts.get("ocsvm") is not None and models_artifacts.get("scaler") is not None and models_artifacts.get("label_encoders") is not None and models_artifacts.get("model_columns") is not None

        col1, col2 = st.columns(2)
        with col1:
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_main_unique_key_v2", disabled=not ae_available)
            if not ae_available: st.caption("Model Autoencoder / artefaknya tidak dapat dimuat.")
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_main_unique_key_v2", disabled=not ocsvm_available)
            if not ocsvm_available: st.caption("Model OC-SVM / artefaknya tidak dapat dimuat.")

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            else:
                with st.spinner("Sedang memproses log... Ini mungkin memakan waktu beberapa saat, mohon tunggu. ⏳"):
                    process_start_time = time.time()
                    try:
                        df_raw_original_with_index = parse_log_file(temp_input_filepath).reset_index() #
                        
                        if df_raw_original_with_index.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                            if os.path.exists(temp_input_filepath): os.remove(temp_input_filepath)
                        else:
                            df_for_preprocessing = df_raw_original_with_index.copy()
                            
                            # Menggunakan fungsi preprocess_dashboard_data yang sudah didefinisikan di atas
                            df_scaled, _ = preprocess_dashboard_data( 
                                df_for_preprocessing,
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
                                st.session_state["uploaded_file_name_for_download"] = uploaded_file.name # Simpan nama file untuk download
                                
                                st.success(f"Parsing dan pra-pemrosesan selesai! Waktu: {time.time() - process_start_time:.2f} detik.", icon="🎉")
                                st.session_state["detection_ready"] = True
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                        st.exception(e) # Tampilkan traceback di Streamlit untuk debug
                    finally:
                        if os.path.exists(temp_input_filepath):
                            try:
                                os.remove(temp_input_filepath)
                            except Exception as e_del:
                                print(f"Gagal menghapus file temporer {temp_input_filepath}: {e_del}")
            st.rerun() # Rerun untuk update tampilan setelah tombol ditekan


    if st.session_state.get("detection_ready", False):
        st.markdown("---")
        st.header("3. Hasil Deteksi")

        df_raw_original = st.session_state.get("df_raw_original_with_index", pd.DataFrame())
        df_scaled = st.session_state.get("df_scaled_for_detection", pd.DataFrame())
        run_ae = st.session_state.get("run_ae_flag", False)
        run_ocsvm = st.session_state.get("run_ocsvm_flag", False)
        uploaded_file_name = st.session_state.get("uploaded_file_name_for_download", "log_yang_diunggah")


        total_records = len(df_raw_original)
        st.metric(label="Total Records Diproses", value=total_records)

        if run_ae and models_artifacts.get("autoencoder") and not df_scaled.empty:
            with st.spinner("Mendeteksi anomali dengan Autoencoder..."):
                ae_anomalies_series, ae_mse_series = get_autoencoder_anomalies(
                    models_artifacts["autoencoder"],
                    df_scaled,
                    training_mse=models_artifacts.get("training_mse_ae")
                )
            
            ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index
            ae_anomalies_count = len(ae_anomalies_indices)
            st.metric(label="Anomali Terdeteksi oleh Autoencoder", value=ae_anomalies_count)

            if ae_anomalies_count > 0:
                st.subheader(f"📜 Log Anomali - Autoencoder ({ae_anomalies_count} Log)")
                anomalous_logs_ae_df = df_raw_original[df_raw_original.index.isin(ae_anomalies_indices)]
                
                for idx in anomalous_logs_ae_df.index:
                    log_dict = anomalous_logs_ae_df.loc[idx].to_dict()
                    log_dict.pop('index', None) 
                    log_str_parts = []
                    for key, value in log_dict.items():
                        if pd.notna(value) and value != '':
                            if isinstance(value, str) and (' ' in value or value == ''):
                                log_str_parts.append(f'{key}="{value}"')
                            else:
                                log_str_parts.append(f'{key}={value}')
                    reconstructed_log_str = " ".join(log_str_parts)
                    st.code(reconstructed_log_str, language="text") # Ganti language ke text untuk log umum
                
                df_ae_anomalies_details = df_raw_original.loc[ae_anomalies_indices].copy()
                # Tambahkan kolom MSE ke df_ae_anomalies_details
                # Pastikan ae_mse_series memiliki indeks yang benar dan sesuai
                if not ae_mse_series.empty and not df_ae_anomalies_details.empty:
                     # Ambil nilai MSE yang sesuai dengan indeks anomali
                    df_ae_anomalies_details['AE_MSE'] = ae_mse_series.loc[ae_anomalies_indices].values


                csv_ae = convert_df_to_csv(df_ae_anomalies_details)
                st.download_button(
                    label="📥 Unduh Detail Anomali AE (CSV)",
                    data=csv_ae,
                    file_name=f"anomalies_AE_{uploaded_file_name}.csv",
                    mime="text/csv",
                    key="download_ae_v2"
                )
            else:
                st.info("Tidak ada anomali terdeteksi oleh Autoencoder.")
            st.markdown("---")

        if run_ocsvm and models_artifacts.get("ocsvm") and not df_scaled.empty:
            with st.spinner("Mendeteksi anomali dengan One-Class SVM..."):
                ocsvm_anomalies_series, ocsvm_scores_series = get_ocsvm_anomalies(
                    models_artifacts["ocsvm"],
                    df_scaled
                )

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
                    st.code(reconstructed_log_str, language="text")

                df_ocsvm_anomalies_details = df_raw_original.loc[ocsvm_anomalies_indices].copy()
                if not ocsvm_scores_series.empty and not df_ocsvm_anomalies_details.empty:
                    df_ocsvm_anomalies_details['OCSVM_Score'] = ocsvm_scores_series.loc[ocsvm_anomalies_indices].values

                csv_ocsvm = convert_df_to_csv(df_ocsvm_anomalies_details)
                st.download_button(
                    label="📥 Unduh Detail Anomali OC-SVM (CSV)",
                    data=csv_ocsvm,
                    file_name=f"anomalies_OCSVM_{uploaded_file_name}.csv",
                    mime="text/csv",
                    key="download_ocsvm_v2"
                )
            else:
                st.info("Tidak ada anomali terdeteksi oleh OC-SVM.")
            st.markdown("---")
        
        st.session_state["detection_ready"] = False 

    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.username = "Penguji Dashboard"
    
    run_dashboard_page()
