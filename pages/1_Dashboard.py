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
import matplotlib.pyplot as plt # Untuk plot histogram
import seaborn as sns # Untuk plot yang lebih baik (opsional, bisa juga hanya matplotlib)

# ... (Impor dari models.py dan konfigurasi path tetap sama seperti sebelumnya) ...
# Impor fungsi dari models.py
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models import parse_log_file, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    # ... (Error handling impor tetap sama) ...
    st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root. Error: {e}")
    st.stop()

# --- Konfigurasi Path --- (Tetap sama)
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


# --- Fungsi Pemuatan Model (load_anomaly_models_and_artifacts - tetap sama) ---
@st.cache_resource
def load_anomaly_models_and_artifacts():
    # ... (Definisi fungsi ini sama seperti pada respons sebelumnya yang sudah memperbaiki nama file) ...
    """Memuat semua model dan artefak yang dibutuhkan."""
    models_artifacts = {"loaded_successfully": True, "messages": []}

    def check_and_load(path, name, load_func, type_name, icon, is_tf_model=False):
        if os.path.exists(path):
            try:
                if is_tf_model:
                    models_artifacts[name] = load_func(path)
                else:
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


# --- Fungsi Konversi CSV (convert_df_to_csv - tetap sama) ---
@st.cache_data 
def convert_df_to_csv(df):
    # ... (Definisi fungsi ini sama) ...
    return df.to_csv(index=False).encode('utf-8')

# --- Fungsi Pra-pemrosesan Data Dashboard (preprocess_dashboard_data - tetap sama) ---
def preprocess_dashboard_data(df_raw, label_encoders_loaded, model_cols_trained, scaler_loaded):
    # ... (Definisi fungsi ini sama seperti pada respons sebelumnya yang sudah memperbaiki indentasi) ...
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    features_to_use = model_cols_trained 

    df_processed_display = df_raw.copy()
    # df_model_input = pd.DataFrame() # Tidak perlu, langsung dari df_processed_display

    missing_cols_for_display = []
    for col in features_to_use:
        if col not in df_processed_display.columns:
            df_processed_display[col] = 'Unknown'
            missing_cols_for_display.append(col)
    # if missing_cols_for_display: # Komentari warning agar tidak terlalu ramai
    #     st.warning(f"Kolom berikut tidak ditemukan di file log input dan diisi dengan 'Unknown': {', '.join(missing_cols_for_display)}")

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
            # st.error(f"Label Encoder untuk kolom '{col}' tidak ditemukan.") # Komentari error
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
    
    if 'models_artifacts_loaded' not in st.session_state:
        st.session_state.models_artifacts_loaded = load_anomaly_models_and_artifacts()
    
    models_artifacts = st.session_state.models_artifacts_loaded
    
    # ... (Expander status pemuatan model tetap sama) ...
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
        if st.button("🔄 Coba Muat Ulang Artefak"):
            if "models_artifacts_loaded" in st.session_state:
                del st.session_state.models_artifacts_loaded
            st.rerun()
        return

    # ... (Bagian Unggah File Log dan Opsi Deteksi tetap sama) ...
    st.markdown("---")
    st.header("1. Unggah File Log Fortigate")
    uploaded_file = st.file_uploader(
        "Pilih file log (.txt atau .log)",
        type=["txt", "log"],
        key="file_uploader_dashboard_v4", # Ganti key
        help="Unggah file log Fortigate Anda dalam format .txt atau .log untuk dianalisis."
    )

    if 'detection_triggered' not in st.session_state:
        st.session_state.detection_triggered = False
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}


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
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_v4", disabled=not ae_available)
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_v4", disabled=not ocsvm_available)

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=critical_artifacts_missing):
            st.session_state.detection_triggered = True
            st.session_state.results_cache = {} # Reset cache hasil sebelumnya
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
                st.session_state.detection_triggered = False
            else:
                with st.spinner("Sedang memproses log..."):
                    # ... (Logika parsing dan pra-pemrosesan df_raw_original_with_index dan df_scaled) ...
                    # --- Ini adalah bagian inti yang perlu disesuaikan dari respons sebelumnya ---
                    df_raw_original_with_index = parse_log_file(temp_input_filepath).reset_index()
                    if not df_raw_original_with_index.empty:
                        df_scaled, _ = preprocess_dashboard_data(
                            df_raw_original_with_index.copy(),
                            models_artifacts.get("label_encoders"),
                            models_artifacts.get("model_columns"),
                            models_artifacts.get("scaler")
                        )
                        if not df_scaled.empty:
                            st.session_state.results_cache["df_raw_original"] = df_raw_original_with_index.drop(columns=['index'], errors='ignore')
                            st.session_state.results_cache["df_raw_original_with_idx"] = df_raw_original_with_index # Simpan dengan index asli untuk pencocokan
                            st.session_state.results_cache["df_scaled"] = df_scaled
                            st.session_state.results_cache["run_ae"] = run_autoencoder
                            st.session_state.results_cache["run_ocsvm"] = run_ocsvm
                            st.session_state.results_cache["uploaded_file_name"] = uploaded_file.name
                            st.success("Parsing dan pra-pemrosesan selesai!")
                        else:
                            st.error("Pra-pemrosesan data gagal.")
                            st.session_state.detection_triggered = False
                    else:
                        st.error("File log kosong atau gagal diparsing.")
                        st.session_state.detection_triggered = False
                    # --- Akhir bagian inti ---
                if os.path.exists(temp_input_filepath):
                    os.remove(temp_input_filepath)
        # Tidak ada st.rerun() di sini agar hasil bisa langsung diproses di bawah
    
    # --- Bagian 3: Hasil Deteksi & Metrik Evaluasi (Tampil jika detection_triggered = True dan ada hasil) ---
    if st.session_state.get("detection_triggered") and st.session_state.results_cache.get("df_scaled") is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi & Metrik Evaluasi")

        results = st.session_state.results_cache
        df_raw_original = results["df_raw_original"]
        df_raw_original_with_idx = results["df_raw_original_with_idx"] # dengan kolom 'index' asli
        df_scaled = results["df_scaled"]
        run_ae = results["run_ae"]
        run_ocsvm = results["run_ocsvm"]
        uploaded_file_name = results["uploaded_file_name"]
        
        total_records = len(df_raw_original)
        
        # --- Metrik Agregat ---
        st.subheader("📈 Ringkasan Deteksi")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Records Diproses", total_records)

        # Inisialisasi untuk menyimpan hasil deteksi
        ae_anomalies_indices = pd.Index([])
        ocsvm_anomalies_indices = pd.Index([])
        
        # Ambil hasil deteksi jika model dijalankan
        if run_ae and models_artifacts.get("autoencoder"):
            ae_anomalies_series, ae_mse_series_current = get_autoencoder_anomalies(
                models_artifacts["autoencoder"], df_scaled, training_mse=models_artifacts.get("training_mse_ae")
            )
            st.session_state.results_cache["ae_mse_series"] = ae_mse_series_current # Simpan untuk evaluasi
            ae_anomalies_indices = ae_anomalies_series[ae_anomalies_series == True].index
            col_m2.metric("Anomali (Autoencoder)", len(ae_anomalies_indices))
        else:
            col_m2.metric("Anomali (Autoencoder)", "N/A" if run_ae else "Tidak Dijalankan")

        if run_ocsvm and models_artifacts.get("ocsvm"):
            ocsvm_anomalies_series, ocsvm_scores_series_current = get_ocsvm_anomalies(
                models_artifacts["ocsvm"], df_scaled
            )
            st.session_state.results_cache["ocsvm_scores_series"] = ocsvm_scores_series_current # Simpan untuk evaluasi
            ocsvm_anomalies_indices = ocsvm_anomalies_series[ocsvm_anomalies_series == True].index
            col_m3.metric("Anomali (OC-SVM)", len(ocsvm_anomalies_indices))
        else:
            col_m3.metric("Anomali (OC-SVM)", "N/A" if run_ocsvm else "Tidak Dijalankan")

        st.markdown("---")

        # --- Evaluasi Model Autoencoder ---
        if run_ae and models_artifacts.get("autoencoder"):
            with st.container(border=True):
                st.subheader("🔍 Evaluasi Model Autoencoder")
                ae_mse_series_current = st.session_state.results_cache.get("ae_mse_series")
                if ae_mse_series_current is not None:
                    st.write("**Reconstruction Error (MSE) untuk Data Unggahan:**")
                    fig_ae, ax_ae = plt.subplots()
                    sns.histplot(ae_mse_series_current, kde=True, ax=ax_ae, bins=50)
                    ax_ae.set_title("Distribusi Reconstruction Error (MSE) - Data Unggahan")
                    ax_ae.set_xlabel("Mean Squared Error (MSE)")
                    ax_ae.set_ylabel("Frekuensi")
                    
                    # Ambil threshold dari training jika ada, atau hitung dari data saat ini (seperti di models.py)
                    threshold_val_ae = np.percentile(models_artifacts.get("training_mse_ae"), 95) if models_artifacts.get("training_mse_ae") is not None and len(models_artifacts.get("training_mse_ae")) > 0 else np.percentile(ae_mse_series_current, 95)
                    ax_ae.axvline(threshold_val_ae, color='r', linestyle='--', label=f'Threshold Anomali ({threshold_val_ae:.4f})')
                    ax_ae.legend()
                    st.pyplot(fig_ae)
                    plt.close(fig_ae) # Tutup figure agar tidak memakan memori

                    st.markdown("""
                    **Penjelasan Reconstruction Error:**
                    * Grafik histogram di atas menunjukkan distribusi *Mean Squared Error* (MSE) atau *Reconstruction Error* dari model Autoencoder untuk setiap log dalam data yang Anda unggah.
                    * MSE mengukur seberapa besar perbedaan antara log asli dan log hasil rekonstruksi oleh Autoencoder setelah melalui proses kompresi (encoding) dan dekompresi (decoding).
                    * **Interpretasi:**
                        * Log dengan MSE **rendah** mirip dengan data normal yang telah dipelajari model.
                        * Log dengan MSE **tinggi** (melewati garis merah/threshold) dianggap sebagai **anomali**, karena model kesulitan merekonstruksinya. Ini menandakan bahwa pola log tersebut berbeda signifikan dari pola normal.
                        * Threshold anomali (garis merah) umumnya ditentukan dari data training (misalnya, persentil ke-95 dari MSE data training normal). Jika MSE data training tidak tersedia, threshold bisa diestimasi dari data saat ini, namun ini kurang ideal.
                    """)
                else:
                    st.info("Data MSE untuk Autoencoder tidak tersedia.")
                
                if not ae_anomalies_indices.empty:
                    st.write(f"**Contoh Log Anomali (Autoencoder):** ({len(ae_anomalies_indices)} terdeteksi)")
                    for idx_scaled in ae_anomalies_indices[:min(5, len(ae_anomalies_indices))]: # Tampilkan maks 5 contoh
                        log_entry_series = df_raw_original_with_idx.loc[idx_scaled]
                        log_dict = log_entry_series.to_dict()
                        log_dict.pop('index', None)
                        log_str_parts = [f'{k}="{v}"' if isinstance(v, str) and (' ' in v or v == '') else f'{k}={v}' for k, v in log_dict.items() if pd.notna(v) and str(v).strip() != '']
                        reconstructed_log_str = " ".join(log_str_parts)
                        with st.expander(f"Log Anomali AE #{idx_scaled + 1} (MSE: {ae_mse_series_current.loc[idx_scaled]:.4f})"):
                            st.code(reconstructed_log_str, language="text")
                    
                    # CSV untuk diunduh (hanya log mentah/parsed fields dari df_raw_original)
                    df_ae_anomalies_for_csv = df_raw_original[df_raw_original.index.isin(ae_anomalies_indices)]
                    csv_ae = convert_df_to_csv(df_ae_anomalies_for_csv)
                    st.download_button(
                        label="📥 Unduh Log Anomali AE (Parsed Fields)", data=csv_ae,
                        file_name=f"anomalies_AE_{uploaded_file_name}.csv", mime="text/csv", key="download_ae_v4"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh Autoencoder pada data ini.")


        # --- Evaluasi Model One-Class SVM ---
        if run_ocsvm and models_artifacts.get("ocsvm"):
            with st.container(border=True):
                st.subheader("🔍 Evaluasi Model One-Class SVM")
                ocsvm_scores_series_current = st.session_state.results_cache.get("ocsvm_scores_series")
                if ocsvm_scores_series_current is not None:
                    st.write("**Distribusi Decision Score untuk Data Unggahan:**")
                    fig_ocsvm, ax_ocsvm = plt.subplots()
                    sns.histplot(ocsvm_scores_series_current, kde=True, ax=ax_ocsvm, bins=50, color="green")
                    ax_ocsvm.set_title("Distribusi Decision Score (OC-SVM) - Data Unggahan")
                    ax_ocsvm.set_xlabel("Decision Score")
                    ax_ocsvm.set_ylabel("Frekuensi")
                    ax_ocsvm.axvline(0, color='r', linestyle='--', label='Threshold Anomali (Score < 0)')
                    ax_ocsvm.legend()
                    st.pyplot(fig_ocsvm)
                    plt.close(fig_ocsvm)

                    st.markdown("""
                    **Penjelasan Decision Score (OC-SVM):**
                    * Grafik histogram di atas menunjukkan distribusi *Decision Score* dari model One-Class SVM untuk setiap log dalam data yang Anda unggah.
                    * Decision Score mengukur jarak data point dari hyperplane (batas keputusan) yang dipelajari model.
                    * **Interpretasi:**
                        * Log dengan skor **positif** berada di dalam batas yang dipelajari dan dianggap **normal**.
                        * Log dengan skor **negatif** (di sebelah kiri garis merah/threshold 0) berada di luar batas dan dianggap sebagai **anomali**. Semakin negatif skornya, semakin dianggap anomali oleh model.
                    """)
                else:
                    st.info("Data Decision Score untuk OC-SVM tidak tersedia.")

                if not ocsvm_anomalies_indices.empty:
                    st.write(f"**Contoh Log Anomali (OC-SVM):** ({len(ocsvm_anomalies_indices)} terdeteksi)")
                    for idx_scaled in ocsvm_anomalies_indices[:min(5, len(ocsvm_anomalies_indices))]: # Tampilkan maks 5 contoh
                        log_entry_series = df_raw_original_with_idx.loc[idx_scaled]
                        log_dict = log_entry_series.to_dict()
                        log_dict.pop('index', None)
                        log_str_parts = [f'{k}="{v}"' if isinstance(v, str) and (' ' in v or v == '') else f'{k}={v}' for k, v in log_dict.items() if pd.notna(v) and str(v).strip() != '']
                        reconstructed_log_str = " ".join(log_str_parts)
                        with st.expander(f"Log Anomali OC-SVM #{idx_scaled + 1} (Score: {ocsvm_scores_series_current.loc[idx_scaled]:.4f})"):
                            st.code(reconstructed_log_str, language="text")
                    
                    df_ocsvm_anomalies_for_csv = df_raw_original[df_raw_original.index.isin(ocsvm_anomalies_indices)]
                    csv_ocsvm = convert_df_to_csv(df_ocsvm_anomalies_for_csv)
                    st.download_button(
                        label="📥 Unduh Log Anomali OC-SVM (Parsed Fields)", data=csv_ocsvm,
                        file_name=f"anomalies_OCSVM_{uploaded_file_name}.csv", mime="text/csv", key="download_ocsvm_v4"
                    )
                else:
                    st.info("Tidak ada anomali spesifik yang terdeteksi oleh OC-SVM pada data ini.")

        # --- Penjelasan Metrik Evaluasi Klasik ---
        with st.container(border=True):
            st.subheader("📖 Penjelasan Metrik Evaluasi Klasik (Membutuhkan Label Ground Truth)")
            st.markdown("""
            Metrik evaluasi klasik seperti **Precision, Recall, F1-Score, dan ROC Curve (AUC)** umumnya digunakan untuk menilai performa model klasifikasi, termasuk deteksi anomali jika kita memiliki data dengan label yang benar (ground truth).

            * **Precision**: Dari semua item yang diprediksi sebagai anomali, berapa banyak yang benar-benar anomali?
                Formula: `True Positives / (True Positives + False Positives)`
                *Relevansi*: Tinggi jika biaya dari *false positive* (salah menandai normal sebagai anomali) tinggi.

            * **Recall (Sensitivity/True Positive Rate)**: Dari semua item yang sebenarnya anomali, berapa banyak yang berhasil dideteksi oleh model?
                Formula: `True Positives / (True Positives + False Negatives)`
                *Relevansi*: Tinggi jika biaya dari *false negative* (gagal mendeteksi anomali) tinggi.

            * **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Memberikan skor tunggal yang menyeimbangkan kedua metrik tersebut.
                Formula: `2 * (Precision * Recall) / (Precision + Recall)`
                *Relevansi*: Berguna jika Anda membutuhkan keseimbangan antara Precision dan Recall.

            * **ROC Curve & AUC (Area Under the Curve)**:
                * **ROC Curve** adalah plot yang menggambarkan kemampuan diagnostik model klasifikasi biner seiring perubahan threshold diskriminasi. Kurva ini memplot True Positive Rate (Recall) terhadap False Positive Rate (`False Positives / (False Positives + True Negatives)`) pada berbagai pengaturan threshold.
                * **AUC** adalah area di bawah kurva ROC. Nilai AUC berkisar dari 0 hingga 1, di mana 1 menunjukkan model yang sempurna, dan 0.5 menunjukkan model yang tidak lebih baik dari tebakan acak.
                *Relevansi*: Memberikan gambaran menyeluruh tentang performa model di semua kemungkinan threshold.

            **Catatan Penting untuk Aplikasi Ini:**
            Karena aplikasi ini dirancang untuk mendeteksi anomali pada file log baru yang **tidak memiliki label ground truth** (yaitu, kita tidak tahu pasti mana log yang normal dan mana yang anomali sebelumnya), maka **nilai Precision, Recall, F1-Score, dan AUC tidak dapat dihitung secara langsung di sini.** Metrik-metrik ini hanya bisa dihitung jika Anda memiliki dataset terpisah yang sudah dilabeli untuk pengujian.
            """)
        
        st.session_state.detection_triggered = False # Reset trigger setelah semua ditampilkan
        # Hapus cache hasil spesifik untuk run berikutnya
        keys_to_delete = ["df_raw_original", "df_raw_original_with_idx", "df_scaled", "run_ae", "run_ocsvm", "uploaded_file_name", "ae_mse_series", "ocsvm_scores_series"]
        for key in keys_to_delete:
            if key in st.session_state.results_cache:
                del st.session_state.results_cache[key]


    elif uploaded_file is None and not critical_artifacts_missing:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")


# Panggil fungsi utama
if __name__ == "__main__":
    # ... (Inisialisasi session_state untuk pengujian langsung tetap sama) ...
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.username = "Penguji Dashboard"
    
    required_states = ["detection_triggered", "results_cache", "models_artifacts_loaded"]
    for state_key in required_states:
        if state_key not in st.session_state:
            if state_key == "results_cache":
                st.session_state[state_key] = {}
            else:
                st.session_state[state_key] = False

    run_dashboard_page()
