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

    df_processed_display = df_raw.copy() # Untuk menyimpan kolom asli yang akan ditampilkan
    
    # Pastikan semua fitur yang dibutuhkan ada, isi dengan 'Unknown' jika tidak
    for col in features_to_use:
        if col not in df_processed_display.columns:
            df_processed_display[col] = 'Unknown' 
    
    df_model_input = df_processed_display[features_to_use].copy() # Data untuk di-encode dan di-scale
    
    # 1. Pastikan semua fitur diperlakukan sebagai string & tangani NaN (sesuai train_script.ipynb)
    for col in features_to_use:
        df_model_input[col] = df_model_input[col].astype(str).fillna('Unknown')

    # 2. Terapkan Label Encoding yang sudah di-load
    for col in features_to_use:
        if col in label_encoders_loaded:
            le = label_encoders_loaded[col]
            df_model_input[col] = df_model_input[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1 # -1 untuk unknown/unseen
            )
            if -1 in df_model_input[col].unique(): # Tangani nilai -1 (unseen)
                try: 
                    unknown_val_enc = le.transform(['Unknown'])[0] # Coba encode 'Unknown' jika ada di training
                    df_model_input[col] = df_model_input[col].replace(-1, unknown_val_enc)
                except ValueError: # Jika 'Unknown' sendiri tidak ada di .classes_ saat training
                    df_model_input[col] = df_model_input[col].replace(-1, 0) # Fallback ke 0 atau nilai default lain
        else:
            df_model_input[col] = 0 # Default jika encoder tidak ada (seharusnya tidak terjadi jika model_columns benar)

    # 3. Terapkan Scaler yang sudah di-load
    try:
        if df_model_input.shape[1] != scaler_loaded.n_features_in_:
            st.error(f"Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan yang diharapkan scaler ({scaler_loaded.n_features_in_}). Fitur yang diharapkan: {features_to_use}")
            return pd.DataFrame(), df_processed_display[features_to_use] # Kembalikan df asli untuk display
        
        scaled_data_values = scaler_loaded.transform(df_model_input)
        df_scaled = pd.DataFrame(scaled_data_values, columns=features_to_use, index=df_model_input.index)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        return pd.DataFrame(), df_processed_display[features_to_use] # Kembalikan df asli untuk display
        
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
        if st.button("🔄 Coba Muat Ulang Artefak", key="reload_artifacts_btn_dash_v3"):
            if "models_artifacts_loaded" in st.session_state:
                del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)",
        type=["txt", "log"],
        key="file_uploader_dashboard_v8", 
        help="Unggah file log Fortigate Anda dalam format .txt atau .log untuk dianalisis."
    )

    if 'detection_output' not in st.session_state:
        st.session_state.detection_output = None


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
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_v8", disabled=not ae_available)
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_v8", disabled=not ocsvm_available)

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_output = None 
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            else:
                with st.spinner("Sedang memproses log... Ini mungkin memakan waktu beberapa saat. ⏳"):
                    output_data = {"uploaded_file_name": uploaded_file.name, 
                                   "run_ae": run_autoencoder, "run_ocsvm": run_ocsvm,
                                   "df_raw_original": None, "df_scaled": None,
                                   "ae_anomalies_series": None, "ae_mse_series": None,
                                   "ocsvm_anomalies_series": None, "ocsvm_scores_series": None
                                  }
                    try:
                        df_raw_original_parsed = parse_log_file(temp_input_filepath) #
                        
                        if df_raw_original_parsed.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                        else:
                            # Simpan df hasil parse dengan index baru (0,1,2,...)
                            output_data["df_raw_original"] = df_raw_original_parsed.copy().reset_index(drop=True) 
                            
                            df_scaled, _ = preprocess_dashboard_data(
                                df_raw_original_parsed.copy(), # Praproses dari data yang baru diparsing
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("model_columns"),
                                models_artifacts.get("scaler")
                            )

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            else:
                                output_data["df_scaled"] = df_scaled
                                
                                if run_autoencoder and models_artifacts.get("autoencoder"):
                                    ae_anomalies_s, ae_mse_s = get_autoencoder_anomalies(
                                        models_artifacts["autoencoder"], df_scaled, 
                                        training_mse=models_artifacts.get("training_mse_ae")
                                    ) #
                                    output_data["ae_anomalies_series"] = ae_anomalies_s
                                    output_data["ae_mse_series"] = ae_mse_s
                                
                                if run_ocsvm and models_artifacts.get("ocsvm"):
                                    ocsvm_anomalies_s, ocsvm_scores_s = get_ocsvm_anomalies(
                                        models_artifacts["ocsvm"], df_scaled
                                    ) #
                                    output_data["ocsvm_anomalies_series"] = ocsvm_anomalies_s
                                    output_data["ocsvm_scores_series"] = ocsvm_scores_s
                                
                                st.session_state.detection_output = output_data
                                st.success("Parsing, pra-pemrosesan, dan deteksi selesai! Gulir ke bawah untuk melihat hasil.")
                                
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                        st.exception(e) 
                    finally:
                        if os.path.exists(temp_input_filepath):
                            try: os.remove(temp_input_filepath)
                            except Exception as e_del: print(f"Gagal menghapus file temporer {temp_input_filepath}: {e_del}")
    
    # --- Bagian 3: Hasil Deteksi & Metrik Evaluasi ---
    if st.session_state.get("detection_output") is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi & Metrik Evaluasi")

        output = st.session_state.detection_output
        df_raw_original = output.get("df_raw_original") 
        uploaded_file_name = output.get("uploaded_file_name", "log_diunggah")
        
        if df_raw_original is None or df_raw_original.empty:
            st.info("Tidak ada data untuk ditampilkan atau proses belum berhasil.")
            return

        total_records = len(df_raw_original)
        
        st.subheader("📈 Ringkasan Deteksi")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records Diproses", total_records)

        # Ambil hasil deteksi dari output dictionary
        ae_anomalies_series = output.get("ae_anomalies_series", pd.Series(dtype='bool'))
        ae_mse_series_current = output.get("ae_mse_series", pd.Series(dtype='float'))
        ocsvm_anomalies_series = output.get("ocsvm_anomalies_series", pd.Series(dtype='bool'))
        ocsvm_scores_series_current = output.get("ocsvm_scores_series", pd.Series(dtype='float'))

        ae_anomalies_indices = pd.Index([])
        if not ae_anomalies_series.empty:
             ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index
        
        ocsvm_anomalies_indices = pd.Index([])
        if not ocsvm_anomalies_series.empty:
            ocsvm_anomalies_indices = ocsvm_anomalies_series[ocsvm_anomalies_series == True].index

        col_m2.metric("Anomali (Autoencoder)", len(ae_anomalies_indices) if output.get("run_ae", False) and "ae_anomalies_series" in output else ("N/A" if output.get("run_ae", False) else "Tidak Dijalankan"))
        col_m3.metric("Anomali (OC-SVM)", len(ocsvm_anomalies_indices) if output.get("run_ocsvm", False) and "ocsvm_anomalies_series" in output else ("N/A" if output.get("run_ocsvm", False) else "Tidak Dijalankan"))
        
        st.markdown("---")

        # --- Evaluasi Model Autoencoder ---
        if output.get("run_ae", False) and models_artifacts.get("autoencoder"):
            with st.container(border=True):
                st.subheader("Autoencoder: Hasil Deteksi & Evaluasi")
                if ae_mse_series_current is not None and not ae_mse_series_current.empty:
                    st.write("**Reconstruction Error (MSE) untuk Data Unggahan:**")
                    fig_ae, ax_ae = plt.subplots()
                    sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Autoencoder")
                    ax_ae.set_xlabel("Mean Squared Error (MSE)")
                    ax_ae.set_ylabel("Frekuensi")
                    
                    # Menentukan threshold AE
                    training_mse_values = models_artifacts.get("training_mse_ae")
                    if training_mse_values is not None and len(training_mse_values) > 0:
                        threshold_val_ae = np.percentile(training_mse_values, 95)
                        threshold_source = "Data Training"
                    elif not ae_mse_series_current.empty:
                        threshold_val_ae = np.percentile(ae_mse_series_current, 95)
                        threshold_source = "Data Unggahan Saat Ini (Fallback)"
                    else:
                        threshold_val_ae = 0 # Atau nilai default lain jika tidak ada data sama sekali
                        threshold_source = "Default (Tidak ada data MSE)"
                    
                    ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f}) dari {threshold_source}')
                    ax_ae.legend()
                    st.pyplot(fig_ae)
                    plt.close(fig_ae)

                    st.markdown("""
                    **Penjelasan Reconstruction Error:**
                    - Error ini mengukur seberapa baik Autoencoder dapat merekonstruksi data input. 
                    - Nilai error yang tinggi (di atas threshold) menunjukkan bahwa data tersebut berbeda dari pola normal yang dipelajari model dan kemungkinan adalah anomali.
                    """)
                else:
                    st.info("Data MSE untuk Autoencoder tidak tersedia (model mungkin tidak dijalankan atau tidak ada hasil).")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali Terdeteksi oleh Autoencoder:** ({len(ae_anomalies_indices)} log)")
                    anomalous_ae_logs_df_display = df_raw_original.loc[ae_anomalies_indices].copy()
                    anomalous_ae_logs_df_display['AE_MSE_Score'] = ae_mse_series_current.loc[ae_anomalies_indices].values
                    st.dataframe(anomalous_ae_logs_df_display, height=300) # Tampilkan sebagai tabel
                    
                    # CSV untuk diunduh (hanya kolom asli dari log yang diparsing)
                    df_ae_anomalies_for_csv = df_raw_original.loc[ae_anomalies_indices]
                    csv_ae = convert_df_to_csv(df_ae_anomalies_for_csv) 
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Tabular, Tanpa Skor)", data=csv_ae,
                        file_name=f"anomalies_AE_tabular_{uploaded_file_name}.csv", mime="text/csv", key="download_ae_tabular_v8"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh Autoencoder pada data ini.")
            st.markdown("---")

        # --- Evaluasi Model One-Class SVM ---
        if output.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("One-Class SVM: Hasil Deteksi & Evaluasi")
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

                    st.markdown("""
                    **Penjelasan Decision Score (OC-SVM):**
                    - Skor ini menunjukkan sejauh mana sebuah data point berada dari batas keputusan yang dipelajari model. 
                    - Skor negatif menandakan anomali. Semakin negatif skornya, semakin dianggap anomali.
                    """)
                else:
                    st.info("Data Decision Score untuk OC-SVM tidak tersedia.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali Terdeteksi oleh OC-SVM:** ({len(ocsvm_anomalies_indices)} log)")
                    anomalous_ocsvm_logs_df_display = df_raw_original.loc[ocsvm_anomalies_indices].copy()
                    anomalous_ocsvm_logs_df_display['OCSVM_Decision_Score'] = ocsvm_scores_series_current.loc[ocsvm_anomalies_indices].values
                    st.dataframe(anomalous_ocsvm_logs_df_display, height=300) 

                    # CSV untuk diunduh (hanya kolom asli dari log yang diparsing)
                    df_ocsvm_anomalies_for_csv = df_raw_original.loc[ocsvm_anomalies_indices]
                    csv_ocsvm = convert_df_to_csv(df_ocsvm_anomalies_for_csv)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Tabular, Tanpa Skor)", data=csv_ocsvm,
                        file_name=f"anomalies_OCSVM_tabular_{uploaded_file_name}.csv", mime="text/csv", key="download_ocsvm_tabular_v8"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh OC-SVM.")
            st.markdown("---")

        # --- Penjelasan Metrik Evaluasi Klasik (Precision, Recall, F1, ROC/AUC) ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik Evaluasi Klasik (Membutuhkan Label Ground Truth)")
            st.markdown("""
            Metrik evaluasi klasik seperti **Precision, Recall, F1-Score, dan ROC Curve (AUC)** umumnya digunakan untuk menilai performa model klasifikasi, termasuk deteksi anomali jika kita memiliki data dengan label yang benar (ground truth).

            - **Precision**: Dari semua item yang diprediksi sebagai anomali oleh model, berapa persentase yang benar-benar anomali?
                - *Formula*: `True Positives / (True Positives + False Positives)`
                - *Relevansi*: Penting jika biaya dari *false positive* (salah menandai normal sebagai anomali) tinggi.

            - **Recall (Sensitivity/True Positive Rate)**: Dari semua item yang sebenarnya anomali, berapa persentase yang berhasil dideteksi oleh model?
                - *Formula*: `True Positives / (True Positives + False Negatives)`
                - *Relevansi*: Penting jika biaya dari *false negative* (gagal mendeteksi anomali yang sebenarnya) tinggi.

            - **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Memberikan skor tunggal yang menyeimbangkan kedua metrik tersebut.
                - *Formula*: `2 * (Precision * Recall) / (Precision + Recall)`
                - *Relevansi*: Berguna jika Anda membutuhkan keseimbangan antara Precision dan Recall.

            - **ROC Curve & AUC (Area Under the Curve)**:
                - **ROC Curve** adalah plot yang menggambarkan kemampuan diagnostik model pada berbagai ambang batas (threshold). Kurva ini memplot True Positive Rate (Recall) terhadap False Positive Rate.
                - **AUC** adalah area di bawah kurva ROC. Nilai AUC berkisar dari 0 hingga 1 (1 sempurna, 0.5 acak).
                - *Relevansi*: Memberikan gambaran menyeluruh tentang performa model di semua kemungkinan threshold.

            **Catatan Penting untuk Aplikasi Ini:**
            Aplikasi ini mendeteksi anomali pada file log baru yang **tidak memiliki label ground truth** (yaitu, kita tidak tahu pasti mana log yang normal dan mana yang anomali sebelumnya). Oleh karena itu, **nilai aktual Precision, Recall, F1-Score, dan AUC tidak dapat dihitung secara langsung di sini.** Metrik-metrik ini hanya bisa dihitung jika Anda memiliki dataset terpisah yang sudah dilabeli untuk pengujian model.
            """)
        
    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")

# Panggil fungsi utama
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.username = "Penguji Dashboard"
    
    if "models_artifacts_loaded" not in st.session_state: 
        st.session_state.models_artifacts_loaded = load_anomaly_models_and_artifacts()
    if "detection_output" not in st.session_state:
        st.session_state.detection_output = None
    
    run_dashboard_page()
