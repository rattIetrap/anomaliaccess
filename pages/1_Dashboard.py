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
    # Pastikan parse_log_file di models.py sudah dimodifikasi untuk menyertakan _raw_log_line_
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

# --- Fungsi Pra-pemrosesan Data untuk Dashboard (Disesuaikan dengan fitur baru) ---
def preprocess_dashboard_data(df_raw_input, label_encoders_loaded, scaler_loaded, 
                              model_columns_for_scaler, # Daftar & urutan kolom yang masuk ke scaler
                              feature_types_loaded):    # Dict {'categorical_original_names': [...], 'numerical_original_names': [...]}
    if df_raw_input.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_for_processing = df_raw_input.copy() 

    categorical_original_names = feature_types_loaded.get('categorical_original_names', [])
    numerical_original_names = feature_types_loaded.get('numerical_original_names', [])
    
    df_cat_processed_pred = pd.DataFrame(index=df_for_processing.index)
    df_num_processed_pred = pd.DataFrame(index=df_for_processing.index)

    # 1. Proses Fitur Kategorikal Asli
    if categorical_original_names:
        for col_cat in categorical_original_names:
            if col_cat not in df_for_processing.columns:
                df_for_processing[col_cat] = 'Unknown'
            
            s = df_for_processing[col_cat].astype(str).fillna('Unknown').replace('', 'Unknown')
            if col_cat in label_encoders_loaded:
                le = label_encoders_loaded[col_cat]
                current_classes = list(le.classes_)
                df_cat_processed_pred[col_cat] = s.apply(
                    lambda x: le.transform([x])[0] if x in current_classes else -1
                )
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
            if col_num not in df_for_processing.columns:
                df_for_processing[col_num] = 0 
                
            s_num = pd.to_numeric(df_for_processing[col_num], errors='coerce')
            # Isi NaN dengan 0 (konsisten dengan training jika median tidak disimpan & dipakai di training)
            # Jika median dari training disimpan, gunakan itu.
            df_num_processed_pred[col_num] = s_num.fillna(0) 

    # 3. Gabungkan Fitur sesuai urutan model_columns_for_scaler
    df_combined_for_scaling = pd.DataFrame(index=df_for_processing.index)
    missing_cols_for_scaler_warning = []
    for col_name_in_scaler_order in model_columns_for_scaler:
        if col_name_in_scaler_order in df_cat_processed_pred.columns:
            df_combined_for_scaling[col_name_in_scaler_order] = df_cat_processed_pred[col_name_in_scaler_order]
        elif col_name_in_scaler_order in df_num_processed_pred.columns:
            df_combined_for_scaling[col_name_in_scaler_order] = df_num_processed_pred[col_name_in_scaler_order]
        else:
            missing_cols_for_scaler_warning.append(col_name_in_scaler_order)
            df_combined_for_scaling[col_name_in_scaler_order] = 0 

    if missing_cols_for_scaler_warning:
        st.warning(f"Kolom untuk scaler: {missing_cols_for_scaler_warning} tidak ditemukan dan diisi 0.")

    if df_combined_for_scaling.empty and model_columns_for_scaler :
         st.warning("DataFrame gabungan untuk scaling kosong.")
         return pd.DataFrame(), df_raw_input # Kembalikan df mentah jika proses gagal

    # 4. Terapkan Scaler
    df_scaled = pd.DataFrame()
    if not df_combined_for_scaling.empty:
        try:
            if df_combined_for_scaling.shape[1] != scaler_loaded.n_features_in_:
                st.error(f"Jumlah fitur input ({df_combined_for_scaling.shape[1]}) tidak cocok dengan scaler ({scaler_loaded.n_features_in_}).")
                return pd.DataFrame(), df_raw_input
            
            scaled_data_values = scaler_loaded.transform(df_combined_for_scaling)
            df_scaled = pd.DataFrame(scaled_data_values, columns=model_columns_for_scaler, index=df_combined_for_scaling.index)
        except Exception as e:
            st.error(f"Error saat scaling data: {e}")
            return pd.DataFrame(), df_raw_input
            
    return df_scaled, df_raw_input # Mengembalikan df_raw_input asli untuk digunakan sebagai basis display

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
        messages_list = models_artifacts.get("messages", [])
        if messages_list: # Pastikan messages_list tidak None
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
        models_artifacts.get("model_columns") and 
        models_artifacts.get("feature_types") 
    )

    if critical_artifacts_missing:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar.", icon="💔")
        if st.button("🔄 Coba Muat Ulang Artefak", key="reload_artifacts_btn_dash_v8"):
            if "models_artifacts_loaded" in st.session_state: del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)", type=["txt", "log"],
        key="file_uploader_dashboard_v14", 
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
        with col1: run_autoencoder = st.checkbox("Autoencoder", value=True, key="cb_ae_v14", disabled=not ae_available)
        with col2: run_ocsvm = st.checkbox("One-Class SVM", value=True, key="cb_ocsvm_v14", disabled=not ocsvm_available)

        if st.button("Proses Log 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_output = None 
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model.", icon="⚠️")
            else:
                with st.spinner("Memproses log... ⏳"):
                    output_data = {
                        "uploaded_file_name": uploaded_file.name, "run_ae": run_autoencoder, "run_ocsvm": run_ocsvm,
                        "df_full_parsed_with_raw_log": None, # Akan berisi semua kolom parsed + _raw_log_line_
                        "df_scaled_for_model": None,
                        "ae_anomalies_series": None, "ae_mse_series": None,
                        "ocsvm_anomalies_series": None, "ocsvm_scores_series": None
                    }
                    try:
                        df_parsed_with_raw_log = parse_log_file(temp_input_filepath).reset_index(drop=True) #
                        
                        if df_parsed_with_raw_log.empty:
                            st.error("Log kosong atau gagal diparsing.", icon="❌")
                        else:
                            output_data["df_full_parsed_with_raw_log"] = df_parsed_with_raw_log.copy()
                            
                            df_for_model_input = df_parsed_with_raw_log.drop(columns=['_raw_log_line_'], errors='ignore')
                            
                            df_scaled, _ = preprocess_dashboard_data(
                                df_for_model_input, 
                                models_artifacts.get("label_encoders"),
                                models_artifacts.get("scaler"),
                                models_artifacts.get("model_columns"), 
                                models_artifacts.get("feature_types")
                            )

                            if df_scaled.empty:
                                st.error("Pra-pemrosesan gagal.", icon="❌")
                            else:
                                output_data["df_scaled_for_model"] = df_scaled
                            if run_autoencoder and models_artifacts.get("autoencoder"):
                                # Tentukan threshold persentil yang baru dan lebih tinggi di sini
                                ae_threshold_percentile = 98 # <-- COBA NAIKKAN NILAI INI. Mulai dengan 98, lalu coba 99 jika perlu.

                                st.info(f"Info: Deteksi Autoencoder akan menggunakan threshold persentil ke-{ae_threshold_percentile}.")

                                ae_anomalies_s, ae_mse_s = get_autoencoder_anomalies(
                                    models_artifacts["autoencoder"], 
                                    df_scaled, 
                                    training_mse=models_artifacts.get("training_mse_ae"),
                                    threshold_percentile=ae_threshold_percentile # <-- Menggunakan nilai baru
                                ) #
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
        df_full_parsed_for_display = output.get("df_full_parsed_with_raw_log") 
        uploaded_file_name = output.get("uploaded_file_name", "log_diunggah")
        
        if df_full_parsed_for_display is None or df_full_parsed_for_display.empty:
            st.info("Tidak ada data untuk ditampilkan.")
            return

        total_records = len(df_full_parsed_for_display)
        
        st.subheader("📈 Ringkasan Deteksi")
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
                    fig_ae, ax_ae = plt.subplots(); sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Autoencoder"); ax_ae.set_xlabel("MSE"); ax_ae.set_ylabel("Frekuensi")
                    training_mse_values = models_artifacts.get("training_mse_ae")
                    threshold_val_ae = 0; threshold_source = "Default (Tidak ada data MSE)"
                    if training_mse_values is not None and len(training_mse_values) > 0:
                        threshold_val_ae = np.percentile(training_mse_values, 95); threshold_source = "Data Training"
                    elif not ae_mse_series_current.empty:
                        threshold_val_ae = np.percentile(ae_mse_series_current, 95); threshold_source = "Data Unggahan (Fallback)"
                    if threshold_source != "Default (Tidak ada data MSE)" : ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold ({threshold_val_ae:.4f}) dari {threshold_source}')
                    ax_ae.legend(); st.pyplot(fig_ae); plt.close(fig_ae) # Penting untuk menutup figure
                    st.markdown("""**Penjelasan Reconstruction Error:** Error ini mengukur seberapa baik Autoencoder dapat merekonstruksi data input. Nilai error yang tinggi (di atas threshold) menunjukkan bahwa data tersebut berbeda dari pola normal yang dipelajari model dan kemungkinan adalah anomali.""")
                else: st.info("Data MSE untuk Autoencoder tidak tersedia.")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali - Autoencoder:** ({len(ae_anomalies_indices)} log)")
                    anomalous_ae_df_display = df_full_parsed_for_display.loc[ae_anomalies_indices].copy()
                    anomalous_ae_df_display['AE_MSE_Score'] = ae_mse_series_current.loc[ae_anomalies_indices].values
                    # Tampilkan semua kolom parsed + _raw_log_line_ + skor
                    st.dataframe(anomalous_ae_df_display, height=300) 
                    
                    # Excel untuk diunduh: semua kolom parsed dari df_full_parsed_for_display (termasuk _raw_log_line_)
                    # TANPA skor MSE
                    df_ae_anomalies_for_excel = df_full_parsed_for_display.loc[ae_anomalies_indices]
                    excel_data_ae = convert_df_to_excel(df_ae_anomalies_for_excel) 
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Excel, Semua Field Parsed + Raw Log)", data=excel_data_ae,
                        file_name=f"anomalies_AE_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        key="download_ae_excel_v14"
                    )
                else:
                    st.info("Tidak ada anomali oleh Autoencoder.")
            st.markdown("---")

        # --- Evaluasi Model One-Class SVM ---
        if output.get("run_ocsvm", False) and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("One-Class SVM: Hasil Deteksi & Evaluasi")
                if ocsvm_scores_series_current is not None and not ocsvm_scores_series_current.empty:
                    st.write("**Distribusi Decision Score untuk Data Unggahan:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots(); sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM)"); ax_ocsvm.set_xlabel("Decision Score"); ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold (< 0 Anomali)'); ax_ocsvm.legend(); st.pyplot(fig_ocsvm); plt.close(fig_ocsvm) # Penting untuk menutup figure
                    st.markdown("""**Penjelasan Decision Score (OC-SVM):** Skor ini menunjukkan jarak data dari batas keputusan. Skor negatif adalah anomali.""")
                else: st.info("Data Decision Score untuk OC-SVM tidak tersedia.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Tabel Log Anomali - OC-SVM:** ({len(ocsvm_anomalies_indices)} log)")
                    anomalous_ocsvm_df_display = df_full_parsed_for_display.loc[ocsvm_anomalies_indices].copy()
                    anomalous_ocsvm_df_display['OCSVM_Decision_Score'] = ocsvm_scores_series_current.loc[ocsvm_anomalies_indices].values
                    st.dataframe(anomalous_ocsvm_df_display, height=300)

                    df_ocsvm_anomalies_for_excel = df_full_parsed_for_display.loc[ocsvm_anomalies_indices]
                    excel_data_ocsvm = convert_df_to_excel(df_ocsvm_anomalies_for_excel)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Excel, Semua Field Parsed + Raw Log)", data=excel_data_ocsvm,
                        file_name=f"anomalies_OCSVM_details_{uploaded_file_name}.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_ocsvm_excel_v14"
                    )
                else:
                    st.info("Tidak ada anomali oleh OC-SVM.")
            st.markdown("---")

        # --- Penjelasan Metrik Evaluasi yang Digunakan & Metrik Klasik ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik dan Pendekatan Evaluasi Model")
            
            st.markdown("""
            Dalam sistem deteksi anomali unsupervised ini, kita tidak memiliki label ground truth (data yang sudah diketahui pasti mana yang normal dan mana yang anomali) untuk data log yang baru diunggah. Oleh karena itu, evaluasi model lebih difokuskan pada bagaimana model membedakan data berdasarkan apa yang telah dipelajarinya dari data training normal.
            """)

            st.markdown("#### 1. Reconstruction Error (MSE) - Model Autoencoder")
            st.markdown("""
            - **Apa itu?**: Autoencoder dilatih untuk merekonstruksi data input "normal" dengan error yang sekecil mungkin. *Mean Squared Error* (MSE) atau *Reconstruction Error* adalah ukuran seberapa besar perbedaan antara log asli dan log hasil rekonstruksi oleh Autoencoder.
            - **Bagaimana Digunakan untuk Deteksi Anomali?**:
                - Log yang memiliki pola mirip dengan data training normal akan memiliki MSE yang rendah.
                - Log yang memiliki pola sangat berbeda (potensi anomali) akan menghasilkan MSE yang tinggi karena model kesulitan merekonstruksinya.
            - **Threshold**: Sebuah nilai ambang (threshold) ditentukan (idealnya dari persentil MSE pada data training normal, atau sebagai fallback dari data saat ini). Log dengan MSE di atas threshold ini ditandai sebagai anomali.
            - **Interpretasi di Dashboard Ini**: Grafik distribusi MSE menunjukkan sebaran error untuk data yang diunggah. Log yang jatuh di area dengan MSE tinggi (melebihi garis threshold) adalah yang dicurigai sebagai anomali oleh Autoencoder. Semakin tinggi MSE, semakin "berbeda" log tersebut dari yang dianggap normal.
            """)

            st.markdown("#### 2. Decision Score - Model One-Class SVM")
            st.markdown("""
            - **Apa itu?**: One-Class SVM (OC-SVM) belajar sebuah batas (boundary) yang melingkupi data normal saat training. *Decision Score* untuk data baru mengukur jarak data tersebut dari batas ini.
            - **Bagaimana Digunakan untuk Deteksi Anomali?**:
                - Log yang berada di dalam atau pada batas yang dipelajari akan mendapatkan skor non-negatif (biasanya positif).
                - Log yang berada di luar batas akan mendapatkan skor negatif.
            - **Threshold**: Secara default, skor kurang dari 0 menandakan anomali.
            - **Interpretasi di Dashboard Ini**: Grafik distribusi Decision Score menunjukkan sebaran skor untuk data yang diunggah. Log dengan skor negatif (di sebelah kiri garis threshold 0) ditandai sebagai anomali oleh OC-SVM. Semakin negatif skornya, semakin dianggap "jauh" dari wilayah normal oleh model.
            """)

            st.markdown("#### 3. Metrik Evaluasi Klasik (Precision, Recall, F1-Score, ROC/AUC)")
            st.markdown("""
            Metrik evaluasi klasik ini sangat berguna untuk menilai performa model klasifikasi secara kuantitatif, **namun mereka memerlukan data yang sudah memiliki label ground truth (informasi benar/salahnya setiap prediksi).**

            - **Precision**: Dari semua yang diprediksi sebagai anomali, berapa banyak yang *benar-benar* anomali?
                - _Formula Umum_: `True Positives / (True Positives + False Positives)`
            - **Recall (Sensitivity)**: Dari semua yang *sebenarnya* anomali, berapa banyak yang berhasil *terdeteksi* oleh model?
                - _Formula Umum_: `True Positives / (True Positives + False Negatives)`
            - **F1-Score**: Rata-rata harmonik dari Precision dan Recall, memberikan keseimbangan antara keduanya.
                - _Formula Umum_: `2 * (Precision * Recall) / (Precision + Recall)`
            - **ROC Curve & AUC**: Menggambarkan kemampuan model dalam membedakan kelas pada berbagai threshold. AUC (Area Under the Curve) yang lebih tinggi menunjukkan performa yang lebih baik.            
            **Catatan Penting untuk Aplikasi Ini:**
            Aplikasi ini mendeteksi anomali pada file log baru yang **tidak memiliki label ground truth**. Oleh karena itu, **nilai aktual Precision, Recall, F1-Score, dan AUC tidak dapat dihitung secara langsung di sini.**
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
