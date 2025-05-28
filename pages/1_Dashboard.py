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

# --- Halaman Dashboard ---
def run_dashboard_page():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop() 

    st.title("🚀 Dashboard Deteksi Anomali Akses Jaringan")
    
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
        models_artifacts.get("model_columns") # model_columns juga penting
    )

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting (Autoencoder, OC-SVM, Scaler, Label Encoders, Model Columns) gagal dimuat. Fungsi deteksi tidak akan bekerja dengan benar. Pastikan semua artefak ada di folder `trained_models_artifacts` dan skrip `train_script.ipynb` sudah dijalankan dengan sukses.", icon="💔")

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", 
        type=["txt", "log"], 
        key="file_uploader_dashboard_main_unique_key", 
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
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_main_unique_key", disabled=not ae_available)
            if not ae_available: st.caption("Model Autoencoder / artefaknya tidak dapat dimuat.")
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_main_unique_key", disabled=not ocsvm_available)
            if not ocsvm_available: st.caption("Model OC-SVM / artefaknya tidak dapat dimuat.")

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            else:
                with st.spinner("Sedang memproses log... Ini mungkin memakan waktu beberapa saat, mohon tunggu. ⏳"):
                    process_start_time = time.time()
                    try:
                        df_raw = parse_log_file(temp_input_filepath)
                        if df_raw.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                            if os.path.exists(temp_input_filepath): os.remove(temp_input_filepath)
                        else:
                            # --- BLOK PRA-PEMROSESAN BARU ---
                            st.write("Melakukan pra-pemrosesan data input...")
                            df_scaled = pd.DataFrame()
                            df_original_for_output = pd.DataFrame()
                            
                            # Ambil artefak yang dibutuhkan
                            label_encoders_loaded = models_artifacts.get("label_encoders")
                            model_columns_trained = models_artifacts.get("model_columns")
                            scaler_loaded = models_artifacts.get("scaler")

                            if not all([label_encoders_loaded, model_columns_trained, scaler_loaded]):
                                st.error("Artefak pra-pemrosesan (Label Encoders, Model Columns, Scaler) tidak lengkap. Proses tidak dapat dilanjutkan.")
                            else:
                                # Kolom yang digunakan saat training (dari model_columns.pkl)
                                features_to_use = model_columns_trained #
                                
                                df_temp_processed = df_raw.copy()
                                df_model_input_display = pd.DataFrame()

                                # Pastikan semua fitur yang dibutuhkan ada dari log mentah
                                for col in features_to_use:
                                    if col not in df_temp_processed.columns:
                                        df_temp_processed[col] = 'Unknown' 
                                df_model_input_display = df_temp_processed[features_to_use].copy() # Untuk ditampilkan
                                df_model_input_for_le = df_temp_processed[features_to_use].copy()


                                # 1. Pastikan semua fitur diperlakukan sebagai string & tangani NaN
                                for col in features_to_use:
                                    df_model_input_for_le[col] = df_model_input_for_le[col].astype(str).fillna('Unknown')

                                # 2. Terapkan Label Encoding yang sudah di-load
                                for col in features_to_use:
                                    if col in label_encoders_loaded:
                                        le = label_encoders_loaded[col]
                                        df_model_input_for_le[col] = df_model_input_for_le[col].apply(
                                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                                        )
                                        if -1 in df_model_input_for_le[col].unique():
                                            try: # Coba transform 'Unknown'
                                                unknown_val_enc = le.transform(['Unknown'])[0]
                                                df_model_input_for_le[col] = df_model_input_for_le[col].replace(-1, unknown_val_enc)
                                            except ValueError: # Jika 'Unknown' tidak ada di classes
                                                df_model_input_for_le[col] = df_model_input_for_le[col].replace(-1, 0) # Fallback ke 0
                                    else:
                                        df_model_input_for_le[col] = 0 # Default jika encoder tidak ada

                                # 3. Terapkan Scaler yang sudah di-load
                                if df_model_input_for_le.shape[1] != scaler_loaded.n_features_in_:
                                    st.error(f"Jumlah fitur untuk scaling ({df_model_input_for_le.shape[1]}) tidak cocok dengan scaler ({scaler_loaded.n_features_in_}). Fitur yg diharapkan: {features_to_use}")
                                else:
                                    scaled_data_values = scaler_loaded.transform(df_model_input_for_le) #
                                    df_scaled = pd.DataFrame(scaled_data_values, columns=features_to_use, index=df_model_input_for_le.index)
                                    df_original_for_output = df_model_input_display # Gunakan kolom asli yang dipilih untuk display
                            # --- AKHIR BLOK PRA-PEMROSESAN BARU ---

                            if df_scaled.empty or df_original_for_output.empty:
                                st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            else:
                                results_df = df_original_for_output.copy()
                                results_df.reset_index(drop=True, inplace=True) # Reset index untuk konkatenasi
                                
                                if run_autoencoder:
                                    ae_anomalies, ae_mse = get_autoencoder_anomalies(
                                        models_artifacts["autoencoder"], 
                                        df_scaled, 
                                        training_mse=models_artifacts.get("training_mse_ae")
                                    ) #
                                    # Pastikan Series memiliki index yang sama sebelum assignment
                                    ae_anomalies.index = results_df.index
                                    ae_mse.index = results_df.index
                                    results_df['is_anomaly_ae'] = ae_anomalies
                                    results_df['reconstruction_error_ae'] = ae_mse
                                    st.info("Deteksi Autoencoder selesai.", icon="🤖")
                                
                                if run_ocsvm:
                                    oc_anomalies, oc_scores = get_ocsvm_anomalies(models_artifacts["ocsvm"], df_scaled) #
                                    oc_anomalies.index = results_df.index
                                    oc_scores.index = results_df.index
                                    results_df['is_anomaly_ocsvm'] = oc_anomalies
                                    results_df['decision_score_ocsvm'] = oc_scores
                                    st.info("Deteksi OC-SVM selesai.", icon="🧩")

                                st.success(f"Pemrosesan log selesai! Waktu: {time.time() - process_start_time:.2f} detik.", icon="🎉")
                                st.session_state["results_df"] = results_df 
                                st.session_state["last_file_name"] = uploaded_file.name 
                                st.session_state["last_unique_id"] = unique_id
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                        print(f"Error processing file: {e}") 
                    finally:
                        if os.path.exists(temp_input_filepath):
                            try:
                                os.remove(temp_input_filepath)
                            except Exception as e_del:
                                print(f"Gagal menghapus file temporer {temp_input_filepath}: {e_del}")
            st.rerun() # Rerun untuk update tampilan setelah tombol ditekan

    if "results_df" in st.session_state and st.session_state["results_df"] is not None and not st.session_state["results_df"].empty:
        st.markdown("---")
        st.header("3. Hasil Deteksi")
        
        results_to_show = st.session_state["results_df"].copy() # Gunakan copy untuk filtering
        
        # Tambahkan kolom Combined_Anomaly jika kedua model dijalankan
        ae_col_exists = 'is_anomaly_ae' in results_to_show.columns
        ocsvm_col_exists = 'is_anomaly_ocsvm' in results_to_show.columns

        if ae_col_exists and ocsvm_col_exists:
            results_to_show['Combined_Anomaly'] = results_to_show['is_anomaly_ae'] | results_to_show['is_anomaly_ocsvm']
        elif ae_col_exists:
            results_to_show['Combined_Anomaly'] = results_to_show['is_anomaly_ae']
        elif ocsvm_col_exists:
            results_to_show['Combined_Anomaly'] = results_to_show['is_anomaly_ocsvm']
        else: # Jika tidak ada model yang dijalankan, atau kolom anomali tidak ada
            results_to_show['Combined_Anomaly'] = False 
            
        st.markdown("**Filter Tampilan Hasil:**")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            show_only_anomalies_ae = st.checkbox("Hanya anomali Autoencoder", key="filter_ae_main_unique", value=False, disabled=not ae_col_exists)
        with col_filter2:
            show_only_anomalies_ocsvm = st.checkbox("Hanya anomali OC-SVM", key="filter_ocsvm_main_unique", value=False, disabled=not ocsvm_col_exists)
        with col_filter3:
            show_only_combined_anomalies = st.checkbox("Hanya anomali gabungan", key="filter_combined_main_unique", value=True, disabled=not ('Combined_Anomaly' in results_to_show.columns and results_to_show['Combined_Anomaly'].any()))


        # Terapkan filter
        filtered_results = results_to_show.copy()
        if show_only_anomalies_ae and ae_col_exists:
            filtered_results = filtered_results[filtered_results['is_anomaly_ae'] == True]
        
        if show_only_anomalies_ocsvm and ocsvm_col_exists:
            filtered_results = filtered_results[filtered_results['is_anomaly_ocsvm'] == True]
        
        if show_only_combined_anomalies and 'Combined_Anomaly' in filtered_results.columns:
            filtered_results = filtered_results[filtered_results['Combined_Anomaly'] == True]
            
        if filtered_results.empty:
            st.info("Tidak ada data yang sesuai dengan filter yang dipilih atau tidak ada anomali yang terdeteksi.", icon="ℹ️")
        else:
            st.dataframe(filtered_results, use_container_width=True, height=400)

        # Tombol download selalu untuk semua hasil (sebelum difilter untuk tampilan)
        csv_data = convert_df_to_csv(results_to_show) 
        
        last_name = st.session_state.get("last_file_name", "log")
        last_id = st.session_state.get("last_unique_id", "hasil")
        output_csv_filename = f"hasil_deteksi_{last_id}_{os.path.splitext(last_name)[0]}.csv"
        
        st.download_button(
            label="Unduh Semua Hasil Deteksi (.csv)",
            data=csv_data,
            file_name=output_csv_filename,
            mime="text/csv",
            use_container_width=True,
            key="download_button_dashboard_main_unique"
        )
    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True 
        st.session_state.username = "Penguji Dashboard"
    
    run_dashboard_page()
