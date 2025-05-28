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
import matplotlib.pyplot as plt
import seaborn as sns

# Impor fungsi dari models.py
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Asumsi /pages/ adalah subfolder dari root
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models import parse_log_file, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root ({project_root}). Error: {e}")
    st.stop()

# --- Konfigurasi Path ---
BASE_DIR = project_root
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path ke Model dan Artefak (disesuaikan dengan output train_script.ipynb)
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl")
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy")

# --- Fungsi Pemuatan Model dengan Cache Streamlit ---
@st.cache_resource
def load_anomaly_models_and_artifacts():
    models_artifacts = {"loaded_successfully": True, "messages": []}

    def check_and_load(path, name, load_func, type_name, icon, is_tf_model=False):
        if os.path.exists(path):
            try:
                models_artifacts[name] = load_func(path)
                models_artifacts["messages"].append(("success", f"{type_name} '{os.path.basename(path)}' berhasil dimuat.", icon))
            except Exception as e:
                full_error_message = f"Gagal memuat {type_name} '{os.path.basename(path)}': {str(e)}"
                print(full_error_message) # Untuk log server
                models_artifacts["messages"].append(("error", full_error_message, "🔥"))
                models_artifacts[name] = None
                models_artifacts["loaded_successfully"] = False
        else:
            models_artifacts["messages"].append(("error", f"File {type_name} tidak ditemukan: {os.path.basename(path)} di {MODEL_ARTIFACTS_FOLDER}", "🚨"))
            models_artifacts[name] = None
            models_artifacts["loaded_successfully"] = False

    check_and_load(AUTOENCODER_MODEL_PATH, "autoencoder", load_model, "Model Autoencoder", "🤖", is_tf_model=True)
    check_and_load(OCSVM_MODEL_PATH, "ocsvm", joblib.load, "Model OC-SVM", "🧩")
    check_and_load(SCALER_PATH, "scaler", joblib.load, "Scaler", "⚙️")
    check_and_load(LABEL_ENCODERS_PATH, "label_encoders", joblib.load, "Label Encoders", "🏷️")
    check_and_load(MODEL_COLUMNS_PATH, "model_columns", joblib.load, "Kolom Model", "📊")

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

# --- Fungsi Pra-pemrosesan Data untuk Dashboard (sesuai train_script.ipynb) ---
def preprocess_dashboard_data(df_raw, label_encoders_loaded, model_cols_trained, scaler_loaded):
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    features_to_use = model_cols_trained

    df_processed_display = df_raw.copy()
    
    missing_cols_for_display = []
    for col in features_to_use:
        if col not in df_processed_display.columns:
            df_processed_display[col] = 'Unknown'
            missing_cols_for_display.append(col)
    
    # Komentari warning agar tidak terlalu ramai di UI, bisa di-log jika perlu
    # if missing_cols_for_display:
    #     st.warning(f"Kolom berikut tidak ditemukan di file log input dan diisi dengan 'Unknown': {', '.join(missing_cols_for_display)}")

    df_model_input = df_processed_display[features_to_use].copy()
    
    for col in features_to_use:
        df_model_input[col] = df_model_input[col].astype(str).fillna('Unknown')

    for col in features_to_use:
        if col in label_encoders_loaded:
            le = label_encoders_loaded[col]
            df_model_input[col] = df_model_input[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1 # -1 untuk unknown/unseen
            )
            if -1 in df_model_input[col].unique():
                try: 
                    unknown_val_enc = le.transform(['Unknown'])[0]
                    df_model_input[col] = df_model_input[col].replace(-1, unknown_val_enc)
                except ValueError: 
                    df_model_input[col] = df_model_input[col].replace(-1, 0) # Fallback
        else:
            # st.error(f"Label Encoder untuk kolom '{col}' tidak ditemukan.") # Komentari error
            df_model_input[col] = 0 # Default

    try:
        if df_model_input.shape[1] != scaler_loaded.n_features_in_:
            st.error(f"Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan yang diharapkan scaler ({scaler_loaded.n_features_in_}). Fitur yang diharapkan: {features_to_use}")
            return pd.DataFrame(), df_processed_display[features_to_use]
        
        scaled_data_values = scaler_loaded.transform(df_model_input)
        df_scaled = pd.DataFrame(scaled_data_values, columns=features_to_use, index=df_model_input.index)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        return pd.DataFrame(), df_processed_display[features_to_use]
        
    return df_scaled, df_processed_display[features_to_use] # Mengembalikan kolom asli yang dipilih untuk display

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

    st.title("🚀 Dashboard Perbandingan Model Deteksi Anomali Akses Jaringan")
    
    # Panggil fungsi load_artifacts dan simpan di session_state jika belum ada
    if 'models_artifacts_loaded' not in st.session_state:
        st.session_state.models_artifacts_loaded = load_anomaly_models_and_artifacts()
    models_artifacts = st.session_state.models_artifacts_loaded
    
    with st.expander("ℹ️ Status Pemuatan Model & Artefak", expanded=not models_artifacts.get("loaded_successfully", True)):
        messages_list = models_artifacts.get("messages")
        if messages_list is not None:
            for type_msg, msg, icon in messages_list:
                if type_msg == "success": st.success(msg, icon=icon)
                elif type_msg == "error": st.error(msg, icon=icon)
                elif type_msg == "warning": st.warning(msg, icon=icon)
        else:
            st.caption("Tidak ada pesan status pemuatan model.")

    critical_artifacts_missing = not (
        models_artifacts.get("autoencoder") and
        models_artifacts.get("ocsvm") and
        models_artifacts.get("scaler") and
        models_artifacts.get("label_encoders") and
        models_artifacts.get("model_columns")
    )

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar.", icon="💔")
        if st.button("🔄 Coba Muat Ulang Artefak"):
            if "models_artifacts_loaded" in st.session_state:
                del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)",
        type=["txt", "log"],
        key="file_uploader_dashboard_v5",
        help="Unggah file log Fortigate Anda dalam format .txt atau .log untuk dianalisis."
    )

    # Inisialisasi session state untuk hasil jika belum ada
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None


    if uploaded_file is not None:
        st.markdown(f"File yang diunggah: `{uploaded_file.name}` (`{uploaded_file.size / 1024:.2f} KB`)")
        temp_input_filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}")
        with open(temp_input_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.header("2. Opsi Deteksi & Proses")
        
        ae_available = models_artifacts.get("autoencoder") is not None
        ocsvm_available = models_artifacts.get("ocsvm") is not None

        col1, col2 = st.columns(2)
        with col1:
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_v5", disabled=not ae_available)
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_v5", disabled=not ocsvm_available)

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_results = None # Reset hasil sebelumnya
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            else:
                with st.spinner("Sedang memproses log..."):
                    results_data = {"uploaded_file_name": uploaded_file.name}
                    try:
                        df_raw_original_parsed = parse_log_file(temp_input_filepath) #
                        
                        if df_raw_original_parsed.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                        else:
                            results_data["df_raw_original"] = df_raw_original_parsed.copy() # Simpan df hasil parse
                            
                            df_scaled, _ = preprocess_dashboard_data( # df_display tidak digunakan langsung di sini
                                df_raw_original_parsed.copy(), # Kirim copy untuk preprocessing
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("model_columns"),
                                models_artifacts.get("scaler")
                            )

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            else:
                                results_data["df_scaled"] = df_scaled
                                
                                if run_autoencoder and models_artifacts.get("autoencoder"):
                                    ae_anomalies_s, ae_mse_s = get_autoencoder_anomalies(
                                        models_artifacts["autoencoder"], df_scaled, 
                                        training_mse=models_artifacts.get("training_mse_ae")
                                    ) #
                                    results_data["ae_anomalies_series"] = ae_anomalies_s
                                    results_data["ae_mse_series"] = ae_mse_s
                                
                                if run_ocsvm and models_artifacts.get("ocsvm"):
                                    ocsvm_anomalies_s, ocsvm_scores_s = get_ocsvm_anomalies(
                                        models_artifacts["ocsvm"], df_scaled
                                    ) #
                                    results_data["ocsvm_anomalies_series"] = ocsvm_anomalies_s
                                    results_data["ocsvm_scores_series"] = ocsvm_scores_s
                                
                                st.session_state.detection_results = results_data
                                st.success("Parsing, pra-pemrosesan, dan deteksi selesai!")
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                        st.exception(e)
                    finally:
                        if os.path.exists(temp_input_filepath):
                            try:
                                os.remove(temp_input_filepath)
                            except Exception as e_del:
                                print(f"Gagal menghapus file temporer {temp_input_filepath}: {e_del}")
            # st.rerun() # Tidak perlu jika semua data hasil ada di session_state


    # --- Bagian 3: Hasil Deteksi & Metrik Evaluasi ---
    if st.session_state.get("detection_results") is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi & Metrik Evaluasi")

        results = st.session_state.detection_results
        df_raw_original = results.get("df_raw_original")
        uploaded_file_name = results.get("uploaded_file_name", "log_diunggah")
        
        if df_raw_original is None or df_raw_original.empty:
            st.info("Tidak ada data untuk ditampilkan.")
            return

        total_records = len(df_raw_original)
        
        st.subheader("📈 Ringkasan Deteksi")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records Diproses", total_records)

        ae_anomalies_series = results.get("ae_anomalies_series", pd.Series(dtype='bool'))
        ae_mse_series_current = results.get("ae_mse_series", pd.Series(dtype='float'))
        ocsvm_anomalies_series = results.get("ocsvm_anomalies_series", pd.Series(dtype='bool'))
        ocsvm_scores_series_current = results.get("ocsvm_scores_series", pd.Series(dtype='float'))

        ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index if not ae_anomalies_series.empty else pd.Index([])
        ocsvm_anomalies_indices = ocsvm_anomalies_series[ocsvm_anomalies_series == True].index if not ocsvm_anomalies_series.empty else pd.Index([])

        col_m2.metric("Anomali (Autoencoder)", len(ae_anomalies_indices) if "ae_anomalies_series" in results else ("N/A" if results.get("run_ae", False) else "Tidak Dijalankan"))
        col_m3.metric("Anomali (OC-SVM)", len(ocsvm_anomalies_indices) if "ocsvm_anomalies_series" in results else ("N/A" if results.get("run_ocsvm", False) else "Tidak Dijalankan"))
        
        st.markdown("---")

        # --- Evaluasi Model Autoencoder ---
        if results.get("run_ae", False) and models_artifacts.get("autoencoder"):
            with st.container(border=True):
                st.subheader("🔍 Evaluasi Model Autoencoder")
                if ae_mse_series_current is not None and not ae_mse_series_current.empty:
                    st.write("**Reconstruction Error (MSE) untuk Data Unggahan:**")
                    fig_ae, ax_ae = plt.subplots()
                    sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE)")
                    ax_ae.set_xlabel("Mean Squared Error (MSE)")
                    ax_ae.set_ylabel("Frekuensi")
                    
                    threshold_val_ae = np.percentile(models_artifacts.get("training_mse_ae"), 95) if models_artifacts.get("training_mse_ae") is not None and len(models_artifacts.get("training_mse_ae")) > 0 else np.percentile(ae_mse_series_current, 95)
                    ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f})')
                    ax_ae.legend()
                    st.pyplot(fig_ae)
                    plt.close(fig_ae)

                    st.markdown("...") # Penjelasan Reconstruction Error
                else:
                    st.info("Data MSE untuk Autoencoder tidak dihasilkan atau tidak ada.")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Contoh Log Anomali (Autoencoder):** ({len(ae_anomalies_indices)} terdeteksi)")
                    # df_raw_original sudah tidak memiliki 'index' tambahan
                    anomalous_ae_logs_df = df_raw_original.loc[ae_anomalies_indices]

                    for original_idx in anomalous_ae_logs_df.index: # original_idx adalah indeks dari df_raw_original
                        log_dict = anomalous_ae_logs_df.loc[original_idx].to_dict()
                        log_str_parts = [f'{k}="{v}"' if isinstance(v, str) and (' ' in v or v == '') else f'{k}={v}' for k, v in log_dict.items() if pd.notna(v) and str(v).strip() != '']
                        reconstructed_log_str = " ".join(log_str_parts)
                        with st.expander(f"Log Anomali AE (Indeks Asli: {original_idx}) (MSE: {ae_mse_series_current.loc[original_idx]:.4f})"):
                            st.code(reconstructed_log_str, language="text")
                    
                    csv_ae = convert_df_to_csv(anomalous_ae_logs_df)
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Parsed Fields)", data=csv_ae,
                        file_name=f"anomalies_AE_{uploaded_file_name}.csv", mime="text/csv", key="download_ae_v5"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh Autoencoder.")


        # --- Evaluasi Model One-Class SVM ---
        if results.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("🔍 Evaluasi Model One-Class SVM")
                if ocsvm_scores_series_current is not None and not ocsvm_scores_series_current.empty:
                    st.write("**Distribusi Decision Score untuk Data Unggahan:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots()
                    sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM)")
                    ax_ocsvm.set_xlabel("Decision Score")
                    ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold (< 0 Anomali)')
                    ax_ocsvm.legend()
                    st.pyplot(fig_ocsvm)
                    plt.close(fig_ocsvm)

                    st.markdown("...") # Penjelasan Decision Score
                else:
                    st.info("Data Decision Score untuk OC-SVM tidak dihasilkan atau tidak ada.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Contoh Log Anomali (OC-SVM):** ({len(ocsvm_anomalies_indices)} terdeteksi)")
                    anomalous_ocsvm_logs_df = df_raw_original.loc[ocsvm_anomalies_indices]
                    for original_idx in anomalous_ocsvm_logs_df.index:
                        log_dict = anomalous_ocsvm_logs_df.loc[original_idx].to_dict()
                        log_str_parts = [f'{k}="{v}"' if isinstance(v, str) and (' ' in v or v == '') else f'{k}={v}' for k, v in log_dict.items() if pd.notna(v) and str(v).strip() != '']
                        reconstructed_log_str = " ".join(log_str_parts)
                        with st.expander(f"Log Anomali OC-SVM (Indeks Asli: {original_idx}) (Score: {ocsvm_scores_series_current.loc[original_idx]:.4f})"):
                            st.code(reconstructed_log_str, language="text")
                    
                    csv_ocsvm = convert_df_to_csv(anomalous_ocsvm_logs_df)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Parsed Fields)", data=csv_ocsvm,
                        file_name=f"anomalies_OCSVM_{uploaded_file_name}.csv", mime="text/csv", key="download_ocsvm_v5"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh OC-SVM.")

        # --- Penjelasan Metrik Evaluasi Klasik ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik Evaluasi Klasik (Membutuhkan Label Ground Truth)")
            st.markdown("""
            Metrik evaluasi klasik seperti **Precision, Recall, F1-Score, dan ROC Curve (AUC)** ...
            *(Penjelasan lengkap seperti pada respons sebelumnya)* ...
            **Catatan Penting untuk Aplikasi Ini:**
            Karena aplikasi ini dirancang untuk mendeteksi anomali pada file log baru yang **tidak memiliki label ground truth** ... nilai Precision, Recall, F1-Score, dan AUC tidak dapat dihitung secara langsung di sini.
            """)
        
        # Hapus hasil dari session_state setelah ditampilkan untuk run berikutnya
        # st.session_state.detection_results = None # Atau reset trigger
        # st.session_state.detection_triggered = False # Reset trigger

    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")


# Panggil fungsi utama
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.username = "Penguji Dashboard"
    
    # Inisialisasi session state penting di awal jika belum ada
    if "models_artifacts_loaded" not in st.session_state: # Penting untuk pemuatan pertama
        st.session_state.models_artifacts_loaded = load_anomaly_models_and_artifacts()
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = None # Atau {}
    
    run_dashboard_page()
