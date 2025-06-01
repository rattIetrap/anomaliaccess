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

# Impor fungsi dari models.py
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

# --- Konfigurasi Path ---
BASE_DIR = project_root
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path ke Model dan Artefak (sesuai dengan train_script.ipynb yang baru)
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "autoencoder_model.keras")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "ocsvm_model.pkl")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "scaler.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "label_encoders.pkl") # Berisi encoder untuk kolom kategorikal asli
MODEL_COLUMNS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "model_columns.pkl") # Daftar kolom input untuk scaler (setelah encoding)
FEATURE_TYPES_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "feature_types.pkl") # Menyimpan daftar nama kolom kategorikal & numerik asli
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy")

# --- Fungsi Pemuatan Model dengan Cache Streamlit ---
@st.cache_resource
def load_anomaly_models_and_artifacts():
    models_artifacts = {"loaded_successfully": True, "messages": []}
    def check_and_load(path, name, load_func, type_name, icon, is_tf_model=False):
        # ... (Implementasi fungsi check_and_load tetap sama seperti sebelumnya) ...
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
    check_and_load(MODEL_COLUMNS_PATH, "model_columns", joblib.load, "Kolom Model (untuk Scaler)", "📊")
    check_and_load(FEATURE_TYPES_PATH, "feature_types", joblib.load, "Tipe Fitur (Asli)", "📋") # Memuat tipe fitur

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

# --- Fungsi Pra-pemrosesan Data untuk Dashboard (Disesuaikan dengan fitur baru) ---
def preprocess_dashboard_data(df_raw, label_encoders_loaded, scaler_loaded, 
                              model_columns_for_scaler, # Daftar & urutan kolom yang masuk ke scaler (dari model_columns.pkl)
                              feature_types_loaded):    # Dict {'categorical_original_names': [...], 'numerical_original_names': [...]}
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    categorical_original_names = feature_types_loaded.get('categorical_original_names', [])
    numerical_original_names = feature_types_loaded.get('numerical_original_names', [])
    
    # Kolom asli yang akan kita proses dan juga tampilkan
    all_original_cols_to_process = sorted(list(set(categorical_original_names + numerical_original_names)))
    
    df_for_processing = df_raw.copy()
    df_for_display = pd.DataFrame(index=df_raw.index) # Untuk menyimpan nilai sebelum encoding/scaling tapi setelah cleaning

    # Pastikan semua kolom yang dibutuhkan ada dan bersihkan
    for col in all_original_cols_to_process:
        if col not in df_for_processing.columns:
            df_for_processing[col] = np.nan # Akan diisi di bawah
            df_for_display[col] = 'Unknown' if col in categorical_original_names else 0
        
        if col in categorical_original_names:
            # Untuk display dan juga sebagai basis untuk encoding
            df_for_processing[col] = df_for_processing[col].astype(str).fillna('Unknown').replace('', 'Unknown')
            df_for_display[col] = df_for_processing[col].copy()
        elif col in numerical_original_names:
            s_num = pd.to_numeric(df_for_processing[col], errors='coerce')
            # Untuk display & input model, NaN diisi 0 (atau median dari training jika disimpan)
            # Asumsikan median dari training sudah implisit dalam scaler atau model jika tidak diisi 0
            df_for_processing[col] = s_num.fillna(0) # Default fill 0 untuk konsistensi
            df_for_display[col] = df_for_processing[col].copy()


    # 1. Proses Fitur Kategorikal (menggunakan nama kolom asli)
    df_cat_processed_pred = pd.DataFrame(index=df_for_processing.index)
    if categorical_original_names:
        for col_cat in categorical_original_names:
            s = df_for_processing[col_cat] # Sudah string dan diisi 'Unknown'
            if col_cat in label_encoders_loaded:
                le = label_encoders_loaded[col_cat]
                current_classes = list(le.classes_)
                df_cat_processed_pred[col_cat] = s.apply(
                    lambda x: le.transform([x])[0] if x in current_classes else -1
                )
                if -1 in df_cat_processed_pred[col_cat].unique():
                    unknown_replacement_val = 0
                    if 'Unknown' in current_classes:
                        unknown_replacement_val = le.transform(['Unknown'])[0]
                    elif len(current_classes) > 0:
                         unknown_replacement_val = le.transform([current_classes[0]])[0] 
                    df_cat_processed_pred[col_cat] = df_cat_processed_pred[col_cat].replace(-1, unknown_replacement_val)
            else:
                df_cat_processed_pred[col_cat] = 0 # Fallback jika encoder tidak ada

    # 2. Proses Fitur Numerik (menggunakan nama kolom asli)
    df_num_processed_pred = pd.DataFrame(index=df_for_processing.index)
    if numerical_original_names:
        for col_num in numerical_original_names:
            df_num_processed_pred[col_num] = df_for_processing[col_num] # Sudah numerik dan diisi NaN nya

    # 3. Gabungkan Fitur sesuai urutan model_columns_for_scaler
    df_combined_for_scaling = pd.DataFrame(index=df_for_processing.index)
    missing_cols_for_scaler = []
    for col_name_in_scaler_order in model_columns_for_scaler:
        if col_name_in_scaler_order in df_cat_processed_pred.columns:
            df_combined_for_scaling[col_name_in_scaler_order] = df_cat_processed_pred[col_name_in_scaler_order]
        elif col_name_in_scaler_order in df_num_processed_pred.columns:
            df_combined_for_scaling[col_name_in_scaler_order] = df_num_processed_pred[col_name_in_scaler_order]
        else:
            missing_cols_for_scaler.append(col_name_in_scaler_order)
            df_combined_for_scaling[col_name_in_scaler_order] = 0 # Fallback

    if missing_cols_for_scaler:
        st.warning(f"Kolom berikut yang ada di 'model_columns.pkl' tidak ditemukan di data yang diproses dan diisi 0: {missing_cols_for_scaler}")

    if df_combined_for_scaling.empty and model_columns_for_scaler :
         st.warning("DataFrame gabungan untuk scaling kosong.")
         return pd.DataFrame(), df_for_display[all_original_cols_to_process] if not df_for_display.empty else pd.DataFrame()

    # 4. Terapkan Scaler
    df_scaled = pd.DataFrame()
    if not df_combined_for_scaling.empty:
        try:
            if df_combined_for_scaling.shape[1] != scaler_loaded.n_features_in_:
                st.error(f"Jumlah fitur input ({df_combined_for_scaling.shape[1]}) tidak cocok dengan scaler ({scaler_loaded.n_features_in_}).")
                return pd.DataFrame(), df_for_display[all_original_cols_to_process] if not df_for_display.empty else pd.DataFrame()
            
            scaled_data_values = scaler_loaded.transform(df_combined_for_scaling)
            df_scaled = pd.DataFrame(scaled_data_values, columns=model_columns_for_scaler, index=df_combined_for_scaling.index)
        except Exception as e:
            st.error(f"Error saat scaling data: {e}")
            return pd.DataFrame(), df_for_display[all_original_cols_to_process] if not df_for_display.empty else pd.DataFrame()
    
    # Kembalikan DataFrame display dengan kolom-kolom asli yang relevan
    df_display_final = df_for_display[all_original_cols_to_process].copy() if not df_for_display.empty else pd.DataFrame()
        
    return df_scaled, df_display_final

# --- Fungsi untuk Konversi DataFrame ke Excel ---
@st.cache_data 
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Anomalies')
    processed_data = output.getvalue()
    return processed_data

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
        models_artifacts.get("model_columns") and # Kolom untuk scaler
        models_artifacts.get("feature_types")     # Tipe fitur asli (kat/num)
    )

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar.", icon="💔")
        if st.button("🔄 Coba Muat Ulang Artefak", key="reload_artifacts_btn_dash_v6"):
            if "models_artifacts_loaded" in st.session_state: del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", type=["txt", "log"],
        key="file_uploader_dashboard_v12", 
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
        with col1: run_autoencoder = st.checkbox("Autoencoder", value=True, key="cb_ae_v12", disabled=not ae_available)
        with col2: run_ocsvm = st.checkbox("One-Class SVM", value=True, key="cb_ocsvm_v12", disabled=not ocsvm_available)

        if st.button("Proses Log 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_output = None 
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model.", icon="⚠️")
            else:
                with st.spinner("Memproses log... ⏳"):
                    output_data = {
                        "uploaded_file_name": uploaded_file.name, "run_ae": run_autoencoder, "run_ocsvm": run_ocsvm,
                        "df_original_cleaned_for_display": None, # Untuk display & download Excel
                        "df_scaled_for_model": None,
                        "ae_anomalies_series": None, "ae_mse_series": None,
                        "ocsvm_anomalies_series": None, "ocsvm_scores_series": None
                    }
                    try:
                        # parse_log_file mengembalikan DataFrame dengan kolom '_raw_log_line_' dan field parsed
                        df_parsed_with_raw_log = parse_log_file(temp_input_filepath).reset_index(drop=True) #
                        
                        if df_parsed_with_raw_log.empty:
                            st.error("Log kosong atau gagal diparsing.", icon="❌")
                        else:
                            # Simpan DataFrame yang sudah diparsing (termasuk _raw_log_line_) untuk digunakan nanti
                            output_data["df_parsed_with_raw_log"] = df_parsed_with_raw_log.copy() 
                            
                            # Pra-pemrosesan untuk model
                            # df_for_model_input tidak perlu menyertakan _raw_log_line_ karena itu bukan fitur model
                            df_for_model_input = df_parsed_with_raw_log.drop(columns=['_raw_log_line_'], errors='ignore')
                            
                            df_scaled, df_display_cleaned_subset = preprocess_dashboard_data(
                                df_for_model_input, # Kirim data tanpa raw log line untuk pra-pemrosesan model
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("scaler"),
                                models_artifacts.get("model_columns"), # Kolom input untuk scaler
                                models_artifacts.get("feature_types")  # Tipe fitur asli (kat/num)
                            )
                            # df_display_cleaned_subset berisi kolom asli (kat & num) yang sudah dibersihkan,
                            # sesuai dengan yang digunakan untuk membuat df_scaled.

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan gagal.", icon="❌")
                            else:
                                output_data["df_scaled_for_model"] = df_scaled
                                # Untuk display dan download, kita gunakan semua kolom dari df_parsed_with_raw_log
                                # yang sudah dibersihkan untuk kolom-kolom fitur.
                                # Kita bisa mengambil semua kolom dari df_parsed_with_raw_log untuk display
                                # dan memastikan kolom fitur yang relevan sudah dibersihkan.
                                
                                # Membersihkan kolom penting pada df_parsed_with_raw_log untuk display akhir
                                model_cols_list = models_artifacts.get("model_columns", [])
                                feature_types = models_artifacts.get("feature_types", {})
                                cat_orig_names = feature_types.get('categorical_original_names', [])
                                num_orig_names = feature_types.get('numerical_original_names', [])
                                all_feature_cols_original = list(set(cat_orig_names + num_orig_names))
                                
                                df_display_final = df_parsed_with_raw_log.copy()
                                for col_key in all_feature_cols_original: # Iterasi pada kolom fitur asli
                                    if col_key in df_display_final.columns:
                                        if col_key in cat_orig_names:
                                            df_display_final[col_key] = df_display_final[col_key].astype(str).fillna('Unknown').replace('', 'Unknown')
                                        elif col_key in num_orig_names:
                                            median_val = pd.to_numeric(df_display_final[col_key], errors='coerce').median()
                                            df_display_final[col_key] = pd.to_numeric(df_display_final[col_key], errors='coerce').fillna(median_val if pd.notna(median_val) else 0)
                                    # else: Kolom fitur tidak ada di log, sudah ditangani di preprocess_dashboard_data untuk df_scaled

                                output_data["df_original_cleaned_for_display"] = df_display_final

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
        df_for_display_and_download = output.get("df_original_cleaned_for_display") 
        uploaded_file_name = output.get("uploaded_file_name", "log_diunggah")
        
        if df_for_display_and_download is None or df_for_display_and_download.empty:
            st.info("Tidak ada data untuk ditampilkan.")
            return

        total_records = len(df_for_display_and_download)
        
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
                if ae_mse_series_current is not None and not ae_mse_series_current.empty:
                    st.write("**Reconstruction Error (MSE) untuk Data Unggahan:**")
                    # ... (Kode plot histogram MSE & penjelasan) ...
                    fig_ae, ax_ae = plt.subplots(); sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Autoencoder"); ax_ae.set_xlabel("MSE"); ax_ae.set_ylabel("Frekuensi")
                    training_mse_values = models_artifacts.get("training_mse_ae")
                    threshold_val_ae = 0; threshold_source = "Default (Tidak ada data MSE)"
                    if training_mse_values is not None and len(training_mse_values) > 0:
                        threshold_val_ae = np.percentile(training_mse_values, 95); threshold_source = "Data Training"
                    elif not ae_mse_series_current.empty:
                        threshold_val_ae = np.percentile(ae_mse_series_current, 95); threshold_source = "Data Unggahan (Fallback)"
                    if threshold_source != "Default (Tidak ada data MSE)" : ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f}) dari {threshold_source}')
                    ax_ae.legend(); st.pyplot(fig_ae); plt.close(fig_ae)
                    st.markdown("""**Penjelasan Reconstruction Error:** ...""")
                else: st.info("Data MSE untuk Autoencoder tidak tersedia.")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali - Autoencoder:** ({len(ae_anomalies_indices)} log)")
                    # Tampilkan semua kolom dari df_for_display_and_download yang anomali + skor MSE
                    anomalous_ae_df_display = df_for_display_and_download.loc[ae_anomalies_indices].copy()
                    anomalous_ae_df_display['AE_MSE_Score'] = ae_mse_series_current.loc[ae_anomalies_indices].values
                    # Pilih kolom yang ingin ditampilkan di tabel dashboard (bisa semua atau subset)
                    # Misalnya, tampilkan semua kolom dari df_for_display_and_download + skor
                    st.dataframe(anomalous_ae_df_display, height=300)
                    
                    # Excel untuk diunduh: semua kolom parsed dari df_for_display_and_download (termasuk _raw_log_line_)
                    # TANPA skor MSE
                    df_ae_anomalies_for_excel = df_for_display_and_download.loc[ae_anomalies_indices]
                    excel_data_ae = convert_df_to_excel(df_ae_anomalies_for_excel) 
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Excel, Tabular Parsed)", data=excel_data_ae,
                        file_name=f"anomalies_AE_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        key="download_ae_excel_v12"
                    )
                else:
                    st.info("Tidak ada anomali oleh Autoencoder.")
            st.markdown("---")

        # --- Evaluasi Model One-Class SVM ---
        if output.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("One-Class SVM: Hasil Deteksi & Evaluasi")
                if ocsvm_scores_series_current is not None and not ocsvm_scores_series_current.empty:
                    # ... (Kode plot histogram Decision Score & penjelasan) ...
                    st.write("**Distribusi Decision Score untuk Data Unggahan:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots(); sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM)"); ax_ocsvm.set_xlabel("Decision Score"); ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold (< 0 Anomali)'); ax_ocsvm.legend(); st.pyplot(fig_ocsvm); plt.close(fig_ocsvm)
                    st.markdown("""**Penjelasan Decision Score (OC-SVM):** ...""")
                else: st.info("Data Decision Score untuk OC-SVM tidak tersedia.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali - OC-SVM:** ({len(ocsvm_anomalies_indices)} log)")
                    anomalous_ocsvm_df_display = df_for_display_and_download.loc[ocsvm_anomalies_indices].copy()
                    anomalous_ocsvm_df_display['OCSVM_Decision_Score'] = ocsvm_scores_series_current.loc[ocsvm_anomalies_indices].values
                    st.dataframe(anomalous_ocsvm_df_display, height=300)

                    df_ocsvm_anomalies_for_excel = df_for_display_and_download.loc[ocsvm_anomalies_indices]
                    excel_data_ocsvm = convert_df_to_excel(df_ocsvm_anomalies_for_excel)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Excel, Tabular Parsed)", data=excel_data_ocsvm,
                        file_name=f"anomalies_OCSVM_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_ocsvm_excel_v12"
                    )
                else:
                    st.info("Tidak ada anomali oleh OC-SVM.")
            st.markdown("---")

        # --- Penjelasan Metrik Evaluasi Klasik ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik Evaluasi Klasik (Membutuhkan Label Ground Truth)")
            # ... (Penjelasan metrik klasik tetap sama) ...
            st.markdown("""
            Metrik evaluasi klasik seperti **Precision, Recall, F1-Score, dan ROC Curve (AUC)** ...
            *(Penjelasan lengkap seperti pada respons sebelumnya)* ...
            **Catatan Penting untuk Aplikasi Ini:** ...
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
