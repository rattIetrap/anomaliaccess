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
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

# Impor fungsi dari models.py
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Pastikan parse_log_file di models.py sudah dimodifikasi untuk menyertakan _raw_log_line_
    from models import parse_log_file, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root ({project_root}). Error: {e}")
    st.stop()

# --- Konfigurasi Path ---
BASE_DIR = project_root
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit')
DATA_FOLDER = os.path.join(BASE_DIR, 'data') # Path untuk menyimpan histori
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True) # Pastikan folder data ada

# Path ke Model dan Artefak
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl") 
FEATURE_TYPES_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "feature_types.pkl") 
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
    check_and_load(MODEL_COLUMNS_PATH, "model_columns", joblib.load, "Kolom Input Scaler", "📊")
    check_and_load(FEATURE_TYPES_PATH, "feature_types", joblib.load, "Tipe Fitur Asli", "📋")

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

# --- Fungsi Pra-pemrosesan Data untuk Dashboard ---
def preprocess_dashboard_data(df_raw_input, label_encoders_loaded, scaler_loaded, 
                              model_columns_for_scaler, 
                              feature_types_loaded):
    if df_raw_input.empty:
        return pd.DataFrame()

    df_for_processing = df_raw_input.copy() 
    categorical_original_names = feature_types_loaded.get('categorical_original_names', [])
    numerical_original_names = feature_types_loaded.get('numerical_original_names', [])
    df_cat_processed_pred = pd.DataFrame(index=df_for_processing.index)
    df_num_processed_pred = pd.DataFrame(index=df_for_processing.index)

    # 1. Proses Fitur Kategorikal Asli
    if categorical_original_names:
        for col_cat in categorical_original_names:
            if col_cat not in df_for_processing.columns: df_for_processing[col_cat] = 'Unknown'
            s = df_for_processing[col_cat].astype(str).fillna('Unknown').replace('', 'Unknown')
            if col_cat in label_encoders_loaded:
                le = label_encoders_loaded[col_cat]
                current_classes = list(le.classes_)
                df_cat_processed_pred[col_cat] = s.apply(lambda x: le.transform([x])[0] if x in current_classes else -1)
                if -1 in df_cat_processed_pred[col_cat].unique():
                    unknown_replacement_val = 0
                    if 'Unknown' in current_classes: unknown_replacement_val = le.transform(['Unknown'])[0]
                    elif len(current_classes) > 0: unknown_replacement_val = le.transform([current_classes[0]])[0] 
                    df_cat_processed_pred[col_cat] = df_cat_processed_pred[col_cat].replace(-1, unknown_replacement_val)
            else:
                df_cat_processed_pred[col_cat] = 0 

    # 2. Proses Fitur Numerik Asli
    if numerical_original_names:
        for col_num in numerical_original_names:
            if col_num not in df_for_processing.columns: df_for_processing[col_num] = 0 
            s_num = pd.to_numeric(df_for_processing[col_num], errors='coerce')
            df_num_processed_pred[col_num] = s_num.fillna(0) 

    # 3. Gabungkan Fitur sesuai urutan model_columns_for_scaler
    df_combined_for_scaling = pd.DataFrame(index=df_for_processing.index)
    missing_cols_for_scaler_warning = []
    for col_name in model_columns_for_scaler:
        if col_name in df_cat_processed_pred.columns: df_combined_for_scaling[col_name] = df_cat_processed_pred[col_name]
        elif col_name in df_num_processed_pred.columns: df_combined_for_scaling[col_name] = df_num_processed_pred[col_name]
        else: missing_cols_for_scaler_warning.append(col_name); df_combined_for_scaling[col_name] = 0 
    
    if missing_cols_for_scaler_warning:
        st.warning(f"Kolom scaler: {missing_cols_for_scaler_warning} tidak ditemukan dan diisi 0.")

    if df_combined_for_scaling.empty and model_columns_for_scaler:
         st.warning("DataFrame gabungan untuk scaling kosong.")
         return pd.DataFrame()

    # 4. Terapkan Scaler
    df_scaled = pd.DataFrame()
    if not df_combined_for_scaling.empty:
        try:
            if df_combined_for_scaling.shape[1] != scaler_loaded.n_features_in_:
                st.error(f"Jumlah fitur input ({df_combined_for_scaling.shape[1]}) tidak cocok dengan scaler ({scaler_loaded.n_features_in_}).")
                return pd.DataFrame()
            scaled_data_values = scaler_loaded.transform(df_combined_for_scaling)
            df_scaled = pd.DataFrame(scaled_data_values, columns=model_columns_for_scaler, index=df_combined_for_scaling.index)
        except Exception as e:
            st.error(f"Error saat scaling data: {e}")
            return pd.DataFrame()
            
    return df_scaled

# --- Fungsi untuk Konversi DataFrame ke Excel ---
@st.cache_data 
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Anomalies')
    return output.getvalue()

# --- Fungsi untuk Menyimpan Hasil Harian ---
def save_summary_to_gsheet(detection_date_str, total_logs, ae_anomaly_count, ocsvm_anomaly_count, username):
    """Menyimpan atau memperbarui ringkasan deteksi harian ke Google Sheets menggunakan gspread."""
    try:
        # --- PERUBAHAN UTAMA: Definisikan Scopes ---
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file'
        ]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        gc = gspread.authorize(creds)
        
        # Buka Spreadsheet dan Worksheet
        spreadsheet = gc.open("DeteksiAnomaliHistory") # Pastikan nama ini benar
        worksheet = spreadsheet.worksheet("History") # Pastikan nama sheet ini benar
        
        detection_date = pd.to_datetime(detection_date_str).date()

        # Baca data yang sudah ada
        existing_data = pd.DataFrame(worksheet.get_all_records())
        if not existing_data.empty:
            existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
            # Hapus entri lama untuk tanggal yang sama jika ada
            existing_data = existing_data[existing_data['date'] != detection_date]
        
        new_entry = pd.DataFrame([{
            'date': detection_date,
            'total_logs': total_logs,
            'anomaly_count_ae': ae_anomaly_count,
            'anomaly_count_ocsvm': ocsvm_anomaly_count,
            'detected_by': username
        }])
        
        # Gabungkan data lama dan entri baru
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        
        # Urutkan berdasarkan tanggal
        updated_df['date'] = pd.to_datetime(updated_df['date'])
        updated_df.sort_values(by='date', inplace=True)
        updated_df['date'] = updated_df['date'].dt.strftime('%Y-%m-%d')
        
        # Tulis kembali seluruh DataFrame ke worksheet
        worksheet.clear()
        set_with_dataframe(worksheet, updated_df, include_index=False, resize=True)
        
        st.success(f"Hasil deteksi untuk tanggal {detection_date.strftime('%Y-%m-%d')} berhasil disimpan ke histori di Google Sheets.")

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet 'DeteksiAnomaliHistory' tidak ditemukan. Pastikan nama sudah benar dan telah dibagikan dengan email service account.")
    except Exception as e:
        st.error(f"Gagal menyimpan hasil ke histori: {e}")

# --- Halaman Dashboard Utama ---
def run_dashboard_page():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop()

    st.title("🚀 Dashboard Deteksi Anomali Akses Jaringan")
    
    if 'models_artifacts_loaded' not in st.session_state:
        st.session_state.models_artifacts_loaded = load_anomaly_models_and_artifacts()
    models_artifacts = st.session_state.models_artifacts_loaded
    
    with st.expander("ℹ️ Status Pemuatan Model & Artefak", expanded=not models_artifacts.get("loaded_successfully", True)):
        messages_list = models_artifacts.get("messages", [])
        if messages_list:
            for type_msg, msg, icon in messages_list:
                if type_msg == "success": st.success(msg, icon=icon)
                elif type_msg == "error": st.error(msg, icon=icon)
                elif type_msg == "warning": st.warning(msg, icon=icon)
    
    critical_artifacts_missing = not (models_artifacts.get("autoencoder") and models_artifacts.get("ocsvm") and models_artifacts.get("scaler") and models_artifacts.get("label_encoders") and models_artifacts.get("model_columns") and models_artifacts.get("feature_types"))

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting gagal dimuat...", icon="💔")
        return

    st.markdown("---")
    st.header("1. Unggah File Log Harian")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", type=["txt", "log"],
        key="file_uploader_dashboard_v15"
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
        with col1: run_autoencoder = st.checkbox("Autoencoder", value=True, key="cb_ae_v15", disabled=not ae_available)
        with col2: run_ocsvm = st.checkbox("One-Class SVM", value=True, key="cb_ocsvm_v15", disabled=not ocsvm_available)

        if st.button("Proses Log 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_output = None 
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model.", icon="⚠️")
            else:
                with st.spinner("Memproses log... ⏳"):
                    output_data = {"uploaded_file_name": uploaded_file.name, "run_ae": run_autoencoder, "run_ocsvm": run_ocsvm}
                    try:
                        df_parsed = parse_log_file(temp_input_filepath).reset_index(drop=True)
                        if df_parsed.empty:
                            st.error("Log kosong atau gagal diparsing.", icon="❌")
                        else:
                            output_data["df_full_parsed"] = df_parsed.copy()
                            df_for_model_input = df_parsed.drop(columns=['_raw_log_line_'], errors='ignore')
                            df_scaled = preprocess_dashboard_data(df_for_model_input, models_artifacts.get("label_encoders"), models_artifacts.get("scaler"), models_artifacts.get("model_columns"), models_artifacts.get("feature_types"))
                            if df_scaled.empty:
                                st.error("Pra-pemrosesan gagal.", icon="❌")
                            else:
                                if run_autoencoder:
                                    ae_anomalies, ae_scores = get_autoencoder_anomalies(models_artifacts["autoencoder"], df_scaled, training_mse=models_artifacts.get("training_mse_ae"))
                                    output_data["ae_anomalies_series"] = ae_anomalies
                                    output_data["ae_mse_series"] = ae_scores
                                if run_ocsvm:
                                    ocsvm_anomalies, ocsvm_scores = get_ocsvm_anomalies(models_artifacts["ocsvm"], df_scaled)
                                    output_data["ocsvm_anomalies_series"] = ocsvm_anomalies
                                    output_data["ocsvm_scores_series"] = ocsvm_scores
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
        df_full_parsed_for_display = output.get("df_full_parsed") 
        uploaded_file_name = output.get("uploaded_file_name", "log_diunggah")
        
        if df_full_parsed_for_display is None or df_full_parsed_for_display.empty:
            st.info("Tidak ada data untuk ditampilkan.")
            return

        total_records = len(df_full_parsed_for_display)
        
        st.subheader("📈 Ringkasan Deteksi")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Records Diproses", total_records)

        ae_anomalies_series = output.get("ae_anomalies_series")
        ae_count = ae_anomalies_series.sum() if ae_anomalies_series is not None else 0
        
        ocsvm_anomalies_series = output.get("ocsvm_anomalies_series")
        ocsvm_count = ocsvm_anomalies_series.sum() if ocsvm_anomalies_series is not None else 0

        col_m2.metric("Anomali (AE)", ae_count if output.get("run_ae") else "N/A")
        col_m3.metric("Anomali (OC-SVM)", ocsvm_count if output.get("run_ocsvm") else "N/A")
        
        # --- PERUBAHAN: Tombol Simpan ke Histori ---
        with col_m4:
            st.write("") 
            st.write("") 
            if st.button("Simpan ke Histori 📜", key="save_history_btn"):
                detection_date_str = df_full_parsed_for_display['date'].iloc[0] if 'date' in df_full_parsed_for_display.columns and not df_full_parsed_for_display.empty else None
                if detection_date_str:
                    current_user = st.session_state.get("username", "unknown_user")
                    save_summary_to_gsheet(detection_date_str, total_records, ae_count, ocsvm_count, current_user)
                else:
                    st.warning("Tidak dapat menentukan tanggal dari log untuk menyimpan ke histori.")
        
        st.markdown("---")

        # --- Evaluasi Model Autoencoder ---
        if output.get("run_ae", False) and models_artifacts.get("autoencoder"):
            with st.container(border=True):
                st.subheader("Autoencoder: Hasil Deteksi & Evaluasi")
                # ... (Isi kontainer evaluasi AE sama seperti sebelumnya) ...
                ae_mse_series_current = output.get("ae_mse_series")
                if ae_mse_series_current is not None and not ae_mse_series_current.empty:
                    st.metric(label="Rata-rata MSE Anomali (AE)", value=f"{ae_mse_series_current[ae_anomalies_series == True].mean():.4f}" if ae_anomalies_series.any() else "N/A")
                    st.write("**Grafik Distribusi Reconstruction Error (MSE):**")
                    fig_ae, ax_ae = plt.subplots(); sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Autoencoder"); ax_ae.set_xlabel("MSE"); ax_ae.set_ylabel("Frekuensi")
                    training_mse_values = models_artifacts.get("training_mse_ae")
                    threshold_val_ae = 0; threshold_source = "Default"
                    if training_mse_values is not None and len(training_mse_values) > 0:
                        threshold_val_ae = np.percentile(training_mse_values, 95); threshold_source = "Data Training"
                    elif not ae_mse_series_current.empty:
                        threshold_val_ae = np.percentile(ae_mse_series_current, 95); threshold_source = "Data Unggahan (Fallback)"
                    if threshold_source != "Default" : ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f})')
                    ax_ae.legend(); st.pyplot(fig_ae); plt.close(fig_ae)
                if ae_anomalies_series.any():
                    st.write(f"**Tabel Log Anomali - Autoencoder:** ({ae_anomalies_series.sum()} log)")
                    anomalous_ae_df_display = df_full_parsed_for_display.loc[ae_anomalies_series == True].copy()
                    anomalous_ae_df_display['AE_MSE_Score'] = ae_mse_series_current.loc[ae_anomalies_series == True]
                    st.dataframe(anomalous_ae_df_display, height=300)
                    excel_data_ae = convert_df_to_excel(df_full_parsed_for_display.loc[ae_anomalies_series == True]) 
                    st.download_button(label="📥 Unduh Log Anomali AE (Excel)", data=excel_data_ae, file_name=f"anomalies_AE_{output['uploaded_file_name']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_ae_excel_final")
                else: st.info("Tidak ada anomali oleh Autoencoder.")
            st.markdown("---")

        # --- Evaluasi Model One-Class SVM ---
        if output.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("One-Class SVM: Hasil Deteksi & Evaluasi")
                # ... (Isi kontainer evaluasi OC-SVM sama seperti sebelumnya) ...
                ocsvm_scores_series_current = output.get("ocsvm_scores_series")
                if ocsvm_scores_series_current is not None and not ocsvm_scores_series_current.empty:
                    st.metric(label="Rata-rata Decision Score Anomali (OC-SVM)", value=f"{ocsvm_scores_series_current[ocsvm_anomalies_series == True].mean():.4f}" if ocsvm_anomalies_series.any() else "N/A")
                    st.write("**Grafik Distribusi Decision Score:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots(); sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM)"); ax_ocsvm.set_xlabel("Decision Score"); ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold (< 0 Anomali)'); ax_ocsvm.legend(); st.pyplot(fig_ocsvm); plt.close(fig_ocsvm)
                if ocsvm_anomalies_series.any():
                    st.write(f"**Tabel Log Anomali - OC-SVM:** ({ocsvm_anomalies_series.sum()} log)")
                    anomalous_ocsvm_df_display = df_full_parsed_for_display.loc[ocsvm_anomalies_series == True].copy()
                    anomalous_ocsvm_df_display['OCSVM_Decision_Score'] = ocsvm_scores_series_current.loc[ocsvm_anomalies_series == True]
                    st.dataframe(anomalous_ocsvm_df_display, height=300)
                    excel_data_ocsvm = convert_df_to_excel(df_full_parsed_for_display.loc[ocsvm_anomalies_series == True])
                    st.download_button(label="📥 Unduh Log Anomali OC-SVM (Excel)", data=excel_data_ocsvm, file_name=f"anomalies_OCSVM_details_{output['uploaded_file_name']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_ocsvm_excel_final")
                else: st.info("Tidak ada anomali oleh OC-SVM.")
            st.markdown("---")

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
