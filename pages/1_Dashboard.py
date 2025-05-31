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
import io # Diperlukan untuk konversi ke Excel

# ... (Impor dari models.py dan Konfigurasi Path tetap sama seperti di respons sebelumnya) ...
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models import parse_log_file, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root ({project_root}). Error: {e}")
    st.stop()

BASE_DIR = project_root
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl")
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy")


# ... (Fungsi load_anomaly_models_and_artifacts tetap sama) ...
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
                print(full_error_message)
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

# ... (Fungsi preprocess_dashboard_data tetap sama, pastikan ia menggunakan model_cols_trained dari artefak) ...
def preprocess_dashboard_data(df_raw, label_encoders_loaded, model_cols_trained, scaler_loaded):
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame() # Kembalikan dua DataFrame kosong
    features_to_use = model_cols_trained
    # Buat salinan dari df_raw yang hanya berisi kolom yang akan diproses atau kolom yang ada di model_cols_trained
    # Ini untuk menghindari modifikasi df_raw asli secara langsung jika df_raw digunakan di tempat lain untuk display
    df_for_model_processing = df_raw.copy()

    # Pastikan semua fitur yang dibutuhkan ada, isi dengan 'Unknown' jika tidak
    # Ini penting untuk df_for_model_processing yang akan di-encode dan di-scale
    for col in features_to_use:
        if col not in df_for_model_processing.columns:
            df_for_model_processing[col] = 'Unknown'
        # Handle NaN dan string kosong SEBELUM astype(str) untuk konsistensi
        df_for_model_processing[col] = df_for_model_processing[col].fillna('Unknown').replace('', 'Unknown')
    
    df_model_input = df_for_model_processing[features_to_use].copy()
    
    for col in features_to_use: # Pastikan semua adalah string sebelum encoding
        df_model_input[col] = df_model_input[col].astype(str)

    for col in features_to_use:
        if col in label_encoders_loaded:
            le = label_encoders_loaded[col]
            current_classes = list(le.classes_)
            df_model_input[col] = df_model_input[col].apply(
                lambda x: le.transform([x])[0] if x in current_classes else -1
            )
            if -1 in df_model_input[col].unique():
                unknown_replacement_val = 0 
                if 'Unknown' in current_classes:
                    unknown_replacement_val = le.transform(['Unknown'])[0]
                elif len(current_classes) > 0:
                    unknown_replacement_val = le.transform([current_classes[0]])[0] 
                df_model_input[col] = df_model_input[col].replace(-1, unknown_replacement_val)
        else:
            df_model_input[col] = 0 
    try:
        if df_model_input.shape[1] != scaler_loaded.n_features_in_:
            st.error(f"Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan scaler ({scaler_loaded.n_features_in_}).")
            # Kembalikan DataFrame kosong dan DataFrame display asli jika ada error
            return pd.DataFrame(), df_raw[features_to_use] if not df_raw.empty and all(c in df_raw for c in features_to_use) else pd.DataFrame()
        
        scaled_data_values = scaler_loaded.transform(df_model_input)
        df_scaled = pd.DataFrame(scaled_data_values, columns=features_to_use, index=df_model_input.index)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        return pd.DataFrame(), df_raw[features_to_use] if not df_raw.empty and all(c in df_raw for c in features_to_use) else pd.DataFrame()
        
    # Untuk display, kita kembalikan kolom-kolom fitur dari df_raw asli (sebelum scaling/encoding)
    # atau dari df_for_model_processing jika kolom asli tidak ada di df_raw
    df_display_output = df_for_model_processing[features_to_use].copy()

    return df_scaled, df_display_output


# ... (Fungsi convert_df_to_excel tetap sama) ...
@st.cache_data 
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Anomalies')
    processed_data = output.getvalue()
    return processed_data

# --- Halaman Dashboard ---
def run_dashboard_page():
    # ... (Kode login check, judul, pemuatan artefak, expander status tetap sama) ...
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
        st.error("Satu atau lebih model/artefak penting gagal dimuat...", icon="💔")
        if st.button("🔄 Coba Muat Ulang Artefak", key="reload_artifacts_btn_dash_v5"): # Key unik
            if "models_artifacts_loaded" in st.session_state: del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    # ... (Bagian Unggah File Log dan Opsi Deteksi tetap sama) ...
    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", type=["txt", "log"],
        key="file_uploader_dashboard_v11", # Key unik
        help="Unggah file log Fortigate Anda..."
    )

    if 'detection_output' not in st.session_state:
        st.session_state.detection_output = None

    if uploaded_file is not None:
        st.markdown(f"File: `{uploaded_file.name}` (`{uploaded_file.size / 1024:.2f} KB`)")
        temp_input_filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}")
        with open(temp_input_filepath, "wb") as f: f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.header("2. Opsi Deteksi & Proses")
        ae_available = models_artifacts.get("autoencoder") is not None
        ocsvm_available = models_artifacts.get("ocsvm") is not None
        col1, col2 = st.columns(2)
        with col1: run_autoencoder = st.checkbox("Autoencoder", value=True, key="cb_ae_v11", disabled=not ae_available)
        with col2: run_ocsvm = st.checkbox("One-Class SVM", value=True, key="cb_ocsvm_v11", disabled=not ocsvm_available)

        if st.button("Proses Log 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_output = None 
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model.", icon="⚠️")
            else:
                with st.spinner("Memproses log... ⏳"):
                    output_data = {
                        "uploaded_file_name": uploaded_file.name, "run_ae": run_autoencoder, "run_ocsvm": run_ocsvm,
                        "df_raw_with_raw_log_column": None, "df_scaled": None, # df_raw_with_raw_log_column akan punya '_raw_log_line_'
                        "ae_anomalies_series": None, "ae_mse_series": None,
                        "ocsvm_anomalies_series": None, "ocsvm_scores_series": None
                    }
                    try:
                        # parse_log_file sekarang mengembalikan DataFrame dengan kolom '_raw_log_line_'
                        df_parsed_with_raw_log = parse_log_file(temp_input_filepath).reset_index(drop=True) #
                        
                        if df_parsed_with_raw_log.empty:
                            st.error("Log kosong atau gagal diparsing.", icon="❌")
                        else:
                            output_data["df_raw_with_raw_log_column"] = df_parsed_with_raw_log.copy()
                            
                            # Pra-pemrosesan untuk model menggunakan kolom yang sesuai dari training
                            # df_raw_for_preprocess tidak perlu menyertakan _raw_log_line_ karena itu bukan fitur model
                            df_for_model_input = df_parsed_with_raw_log.drop(columns=['_raw_log_line_'], errors='ignore')
                            
                            df_scaled, _ = preprocess_dashboard_data(
                                df_for_model_input, # Kirim data tanpa raw log line untuk pra-pemrosesan model
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("model_columns"), # Ini adalah fitur model
                                models_artifacts.get("scaler")
                            )

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan gagal.", icon="❌")
                            else:
                                output_data["df_scaled"] = df_scaled
                                # ... (Deteksi anomali AE dan OCSVM tetap sama, menyimpan series ke output_data) ...
                                if run_autoencoder and models_artifacts.get("autoencoder"):
                                    ae_anomalies_s, ae_mse_s = get_autoencoder_anomalies(models_artifacts["autoencoder"], df_scaled, training_mse=models_artifacts.get("training_mse_ae"))
                                    output_data["ae_anomalies_series"] = ae_anomalies_s
                                    output_data["ae_mse_series"] = ae_mse_s
                                if run_ocsvm and models_artifacts.get("ocsvm"):
                                    ocsvm_anomalies_s, ocsvm_scores_s = get_ocsvm_anomalies(models_artifacts["ocsvm"], df_scaled)
                                    output_data["ocsvm_anomalies_series"] = ocsvm_anomalies_s
                                    output_data["ocsvm_scores_series"] = ocsvm_scores_s
                                
                                st.session_state.detection_output = output_data
                                st.success("Proses selesai! Lihat hasil di bawah.")
                                
                    except Exception as e:
                        st.error(f"Error: {e}", icon="🔥"); st.exception(e) 
                    finally:
                        if os.path.exists(temp_input_filepath):
                            try: os.remove(temp_input_filepath)
                            except Exception as e_del: print(f"Gagal hapus temp: {e_del}")
    
    # --- Bagian 3: Hasil Deteksi & Metrik Evaluasi ---
    if st.session_state.get("detection_output") is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi & Metrik Evaluasi")

        output = st.session_state.detection_output
        # df_raw_with_raw_log_column adalah DataFrame yang memiliki kolom '_raw_log_line_' dan field parsed lainnya
        df_raw_with_raw_log_column = output.get("df_raw_with_raw_log_column") 
        uploaded_file_name = output.get("uploaded_file_name", "log_diunggah")
        
        if df_raw_with_raw_log_column is None or df_raw_with_raw_log_column.empty:
            st.info("Tidak ada data untuk ditampilkan.")
            return

        total_records = len(df_raw_with_raw_log_column)
        st.subheader("📈 Ringkasan Deteksi")
        # ... (Kode metrik agregat tetap sama) ...
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records Diproses", total_records)
        ae_anomalies_series = output.get("ae_anomalies_series", pd.Series(dtype='bool'))
        ae_mse_series_current = output.get("ae_mse_series", pd.Series(dtype='float'))
        ocsvm_anomalies_series = output.get("ocsvm_anomalies_series", pd.Series(dtype='bool'))
        ocsvm_scores_series_current = output.get("ocsvm_scores_series", pd.Series(dtype='float'))
        ae_anomalies_indices = pd.Index([])
        if not ae_anomalies_series.empty: ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index
        ocsvm_anomalies_indices = pd.Index([])
        if not ocsvm_anomalies_series.empty: ocsvm_anomalies_indices = ocsvm_anomalies_series[ocsvm_anomalies_series == True].index
        col_m2.metric("Anomali (AE)", len(ae_anomalies_indices) if output.get("run_ae", False) and "ae_anomalies_series" in output else ("N/A" if output.get("run_ae", False) else "Tidak Dijalankan"))
        col_m3.metric("Anomali (OC-SVM)", len(ocsvm_anomalies_indices) if output.get("run_ocsvm", False) and "ocsvm_anomalies_series" in output else ("N/A" if output.get("run_ocsvm", False) else "Tidak Dijalankan"))

        st.markdown("---")

        # --- Evaluasi Model Autoencoder ---
        if output.get("run_ae", False) and models_artifacts.get("autoencoder"):
            with st.container(border=True):
                st.subheader("Autoencoder: Hasil Deteksi & Evaluasi")
                # ... (Kode plot histogram MSE & penjelasan Reconstruction Error tetap sama) ...
                if ae_mse_series_current is not None and not ae_mse_series_current.empty:
                    st.write("**Reconstruction Error (MSE) untuk Data Unggahan:**")
                    fig_ae, ax_ae = plt.subplots(); sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Autoencoder"); ax_ae.set_xlabel("MSE"); ax_ae.set_ylabel("Frekuensi")
                    training_mse_values = models_artifacts.get("training_mse_ae")
                    threshold_val_ae = 0 
                    if training_mse_values is not None and len(training_mse_values) > 0:
                        threshold_val_ae = np.percentile(training_mse_values, 95); threshold_source = "Data Training"
                    elif not ae_mse_series_current.empty:
                        threshold_val_ae = np.percentile(ae_mse_series_current, 95); threshold_source = "Data Unggahan (Fallback)"
                    else: threshold_source = "Default (Tiada MSE)"
                    if threshold_val_ae > 0 : ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f}) dari {threshold_source}')
                    ax_ae.legend(); st.pyplot(fig_ae); plt.close(fig_ae)
                    st.markdown("""**Penjelasan Reconstruction Error:** Error ini mengukur seberapa baik Autoencoder dapat merekonstruksi data input. Nilai error yang tinggi (di atas threshold) menunjukkan bahwa data tersebut berbeda dari pola normal yang dipelajari model dan kemungkinan adalah anomali.""")
                else: st.info("Data MSE untuk Autoencoder tidak tersedia.")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Log Anomali Terdeteksi oleh Autoencoder:** ({len(ae_anomalies_indices)} log)")
                    anomalous_ae_logs_display = df_raw_with_raw_log_column.loc[ae_anomalies_indices, ['_raw_log_line_']].copy()
                    anomalous_ae_logs_display['AE_MSE_Score'] = ae_mse_series_current.loc[ae_anomalies_indices].values
                    
                    # Tampilkan log mentah dan skornya
                    for idx in anomalous_ae_logs_display.index:
                        raw_log = anomalous_ae_logs_display.loc[idx, '_raw_log_line_']
                        score = anomalous_ae_logs_display.loc[idx, 'AE_MSE_Score']
                        with st.expander(f"Log Anomali AE (Indeks: {idx}) (MSE: {score:.4f})"):
                            st.code(raw_log, language="text")
                    
                    # Excel untuk diunduh: semua field parsed + kolom _raw_log_line_
                    df_ae_anomalies_for_excel = df_raw_with_raw_log_column.loc[ae_anomalies_indices].copy()
                    excel_data_ae = convert_df_to_excel(df_ae_anomalies_for_excel) 
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Excel, Tabular Parsed + Raw Log)", data=excel_data_ae,
                        file_name=f"anomalies_AE_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        key="download_ae_excel_v11"
                    )
                else:
                    st.info("Tidak ada anomali oleh Autoencoder.")
            st.markdown("---")

        # --- Evaluasi Model One-Class SVM ---
        if output.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("One-Class SVM: Hasil Deteksi & Evaluasi")
                # ... (Kode plot histogram Decision Score & penjelasan tetap sama) ...
                if ocsvm_scores_series_current is not None and not ocsvm_scores_series_current.empty:
                    st.write("**Distribusi Decision Score untuk Data Unggahan:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots(); sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM)"); ax_ocsvm.set_xlabel("Decision Score"); ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold (< 0 Anomali)'); ax_ocsvm.legend(); st.pyplot(fig_ocsvm); plt.close(fig_ocsvm)
                    st.markdown("""**Penjelasan Decision Score (OC-SVM):** Skor ini menunjukkan jarak data dari batas keputusan. Skor negatif adalah anomali.""")
                else: st.info("Data Decision Score untuk OC-SVM tidak tersedia.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Log Anomali Terdeteksi oleh OC-SVM:** ({len(ocsvm_anomalies_indices)} log)")
                    anomalous_ocsvm_logs_display = df_raw_with_raw_log_column.loc[ocsvm_anomalies_indices, ['_raw_log_line_']].copy()
                    anomalous_ocsvm_logs_display['OCSVM_Decision_Score'] = ocsvm_scores_series_current.loc[ocsvm_anomalies_indices].values

                    for idx in anomalous_ocsvm_logs_display.index:
                        raw_log = anomalous_ocsvm_logs_display.loc[idx, '_raw_log_line_']
                        score = anomalous_ocsvm_logs_display.loc[idx, 'OCSVM_Decision_Score']
                        with st.expander(f"Log Anomali OC-SVM (Indeks: {idx}) (Score: {score:.4f})"):
                            st.code(raw_log, language="text")
                    
                    # Excel untuk diunduh: semua field parsed + kolom _raw_log_line_
                    df_ocsvm_anomalies_for_excel = df_raw_with_raw_log_column.loc[ocsvm_anomalies_indices].copy()
                    excel_data_ocsvm = convert_df_to_excel(df_ocsvm_anomalies_for_excel)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Excel, Tabular Parsed + Raw Log)", data=excel_data_ocsvm,
                        file_name=f"anomalies_OCSVM_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_ocsvm_excel_v11"
                    )
                else:
                    st.info("Tidak ada anomali oleh OC-SVM.")
            st.markdown("---")

        # --- Penjelasan Metrik Evaluasi Klasik (Precision, Recall, F1, ROC/AUC) ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik Evaluasi Klasik (Membutuhkan Label Ground Truth)")
            # ... (Penjelasan metrik klasik tetap sama seperti pada respons sebelumnya) ...
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
