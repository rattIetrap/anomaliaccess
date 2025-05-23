# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import time # Untuk simulasi loading
import joblib
from tensorflow.keras.models import load_model

# Impor fungsi dari models.py (asumsikan models.py ada di direktori root proyek)
# Untuk mengimpor dari direktori root, kita mungkin perlu menyesuaikan sys.path
import sys
# Dapatkan path ke direktori root proyek (satu level di atas folder 'pages')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models import parse_log_file, preprocess_data, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    st.error(f"Gagal mengimpor modul 'models'. Pastikan 'models.py' ada di direktori root. Error: {e}")
    st.stop() # Hentikan eksekusi jika modul penting tidak bisa diimpor

# --- Konfigurasi Path ---
BASE_DIR = project_root # Sekarang BASE_DIR adalah root proyek
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_streamlit') # Folder berbeda untuk Streamlit
# DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads_streamlit') # Tidak digunakan lagi karena download button Streamlit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Path ke Model dan Artefak
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "trained_autoencoder_model.h5")
OCSVM_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "trained_ocsvm_model.joblib")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "trained_scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "trained_label_encoders.joblib")
TRAINING_MSE_AE_PATH = os.path.join(MODEL_ARTIFACTS_FOLDER, "training_mse_ae.npy")


# --- Fungsi Pemuatan Model dengan Cache Streamlit ---
@st.cache_resource # Cache resource agar model tidak dimuat ulang setiap interaksi
def load_anomaly_models_and_artifacts():
    """Memuat semua model dan artefak yang dibutuhkan."""
    models_artifacts = {"loaded_successfully": True, "messages": []}
    try:
        if os.path.exists(AUTOENCODER_MODEL_PATH):
            models_artifacts["autoencoder"] = load_model(AUTOENCODER_MODEL_PATH)
            models_artifacts["messages"].append(("success", f"Model Autoencoder '{os.path.basename(AUTOENCODER_MODEL_PATH)}' berhasil dimuat.", "🤖"))
        else:
            models_artifacts["messages"].append(("error", f"File model Autoencoder tidak ditemukan: {os.path.basename(AUTOENCODER_MODEL_PATH)}", "🚨"))
            models_artifacts["autoencoder"] = None
            models_artifacts["loaded_successfully"] = False

        if os.path.exists(OCSVM_MODEL_PATH):
            models_artifacts["ocsvm"] = joblib.load(OCSVM_MODEL_PATH)
            models_artifacts["messages"].append(("success", f"Model OC-SVM '{os.path.basename(OCSVM_MODEL_PATH)}' berhasil dimuat.", "🧩"))
        else:
            models_artifacts["messages"].append(("error", f"File model OC-SVM tidak ditemukan: {os.path.basename(OCSVM_MODEL_PATH)}", "🚨"))
            models_artifacts["ocsvm"] = None
            models_artifacts["loaded_successfully"] = False
            
        if os.path.exists(SCALER_PATH):
            models_artifacts["scaler"] = joblib.load(SCALER_PATH)
            models_artifacts["messages"].append(("success", f"Scaler '{os.path.basename(SCALER_PATH)}' berhasil dimuat.", "⚙️"))
        else:
            models_artifacts["messages"].append(("error", f"File scaler tidak ditemukan: {os.path.basename(SCALER_PATH)}", "🚨"))
            models_artifacts["scaler"] = None
            models_artifacts["loaded_successfully"] = False

        if os.path.exists(LABEL_ENCODERS_PATH):
            models_artifacts["label_encoders"] = joblib.load(LABEL_ENCODERS_PATH)
            models_artifacts["messages"].append(("success", f"Label Encoders '{os.path.basename(LABEL_ENCODERS_PATH)}' berhasil dimuat.", "🏷️"))
        else:
            models_artifacts["messages"].append(("error", f"File label encoders tidak ditemukan: {os.path.basename(LABEL_ENCODERS_PATH)}", "🚨"))
            models_artifacts["label_encoders"] = None
            models_artifacts["loaded_successfully"] = False
            
        if os.path.exists(TRAINING_MSE_AE_PATH):
            models_artifacts["training_mse_ae"] = np.load(TRAINING_MSE_AE_PATH)
            models_artifacts["messages"].append(("success", f"Training MSE AE '{os.path.basename(TRAINING_MSE_AE_PATH)}' berhasil dimuat.", "📊"))
        else:
            models_artifacts["messages"].append(("warning", f"File training MSE Autoencoder tidak ditemukan: {os.path.basename(TRAINING_MSE_AE_PATH)}. Threshold AE akan dihitung dari data input.", "⚠️"))
            models_artifacts["training_mse_ae"] = None 

    except Exception as e:
        models_artifacts["messages"].append(("error", f"Error saat memuat model atau artefak: {e}", "🔥"))
        models_artifacts["loaded_successfully"] = False
    
    return models_artifacts

# --- Fungsi untuk Konversi DataFrame ke CSV (untuk tombol download) ---
@st.cache_data # Cache data agar konversi tidak dilakukan berulang kali jika data sama
def convert_df_to_csv(df):
    # Penting: Jangan sertakan index jika tidak diinginkan di file CSV
    return df.to_csv(index=False).encode('utf-8')

# --- Halaman Dashboard ---
def run_dashboard_page():
    # st.set_page_config tidak bisa dipanggil di halaman sekunder, hanya di skrip utama.
    # Konfigurasi halaman akan diambil dari streamlit_app.py.
    
    # Cek status login dari session state utama
    if not st.session_state.get("logged_in", False):
        st.warning("Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop() # Hentikan eksekusi jika tidak login

    st.title("🚀 Dashboard Deteksi Anomali Akses")
    # st.markdown(f"Login sebagai: **{st.session_state.get('username', 'Pengguna')}**!") # Sudah ada di sidebar

    # Muat model dan tampilkan pesan pemuatan
    # Pesan loading akan ditampilkan oleh Streamlit saat fungsi cache berjalan pertama kali
    models_artifacts = load_anomaly_models_and_artifacts()
    
    with st.expander("Status Pemuatan Model & Artefak", expanded=not models_artifacts["loaded_successfully"]):
        for type, msg, icon in models_artifacts.get("messages", []):
            if type == "success":
                st.success(msg, icon=icon)
            elif type == "error":
                st.error(msg, icon=icon)
            elif type == "warning":
                st.warning(msg, icon=icon)

    if not models_artifacts["loaded_successfully"]:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar. Silakan periksa path dan file artefak.", icon="💔")
    
    # Tombol logout di sidebar (jika belum ada di streamlit_app.py)
    # Sebaiknya konsisten, jika sudah ada di streamlit_app.py, tidak perlu di sini.
    # if st.sidebar.button("Logout", key="dashboard_logout_button_sidebar", use_container_width=True):
    #     st.session_state["logged_in"] = False
    #     st.session_state["username"] = None
    #     if "login_error" in st.session_state: del st.session_state["login_error"]
    #     if "results_df" in st.session_state: del st.session_state["results_df"]
    #     st.switch_page("streamlit_app.py")

    st.markdown("---")
    st.header("1. Unggah File Log")
    uploaded_file = st.file_uploader(
        "Pilih file log Fortigate (.txt atau .log)", 
        type=["txt", "log"], 
        key="file_uploader_dashboard",
        help="Unggah file log Anda dalam format .txt atau .log untuk dianalisis."
    )

    if uploaded_file is not None:
        # Tampilkan nama file dan ukuran untuk konfirmasi
        st.write(f"File yang diunggah: `{uploaded_file.name}` ({uploaded_file.size / 1024:.2f} KB)")
        
        # Simpan file yang diunggah sementara untuk diproses
        unique_id = str(uuid.uuid4().hex[:8])
        temp_input_filename = f"{unique_id}_{uploaded_file.name}"
        temp_input_filepath = os.path.join(UPLOAD_FOLDER, temp_input_filename)
        
        with open(temp_input_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.header("2. Pilih Model & Proses")
        
        # Cek apakah model yang diperlukan tersedia sebelum menampilkan checkbox
        ae_available = models_artifacts.get("autoencoder") is not None and models_artifacts.get("scaler") is not None
        ocsvm_available = models_artifacts.get("ocsvm") is not None and models_artifacts.get("scaler") is not None

        col1, col2 = st.columns(2)
        with col1:
            run_autoencoder = st.checkbox("Autoencoder", value=True, key="cb_ae_dashboard", disabled=not ae_available)
            if not ae_available:
                st.caption("Model Autoencoder tidak tersedia")
        with col2:
            run_ocsvm = st.checkbox("One-Class SVM", value=True, key="cb_ocsvm_dashboard", disabled=not ocsvm_available)
            if not ocsvm_available:
                st.caption("Model OC-SVM tidak tersedia")

        if st.button("Proses Log Sekarang", type="primary", use_container_width=True, disabled=not models_artifacts["loaded_successfully"]):
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi.", icon="⚠️")
            elif (run_autoencoder and not ae_available) or (run_ocsvm and not ocsvm_available):
                st.error("Satu atau lebih model yang dipilih tidak tersedia. Periksa status pemuatan model.", icon="🚫")
            else:
                with st.spinner("Sedang memproses log... Ini mungkin memakan waktu beberapa saat. ⏳"):
                    time.sleep(0.5) # Sedikit delay untuk UX
                    try:
                        df_raw = parse_log_file(temp_input_filepath)
                        if df_raw.empty:
                            st.error("File log kosong atau gagal diparsing setelah diunggah.", icon="❌")
                            # Hapus file temp jika ada
                            if os.path.exists(temp_input_filepath): os.remove(temp_input_filepath)
                            return 

                        df_scaled, _, _, _, df_original_for_output = preprocess_data(
                            df_raw.copy(), 
                            scaler=models_artifacts.get("scaler"), 
                            label_encoders=models_artifacts.get("label_encoders"), 
                            is_training=False
                        )

                        if df_scaled is None or df_scaled.empty or df_original_for_output is None or df_original_for_output.empty:
                            st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            if os.path.exists(temp_input_filepath): os.remove(temp_input_filepath)
                            return

                        results_df = df_original_for_output.copy()
                        
                        if run_autoencoder:
                            ae_anomalies, ae_mse = get_autoencoder_anomalies(
                                models_artifacts["autoencoder"], 
                                df_scaled, 
                                training_mse=models_artifacts.get("training_mse_ae")
                            )
                            results_df['is_anomaly_ae'] = ae_anomalies
                            results_df['reconstruction_error_ae'] = ae_mse
                            st.info("Deteksi Autoencoder selesai.", icon="🤖")
                        
                        if run_ocsvm:
                            oc_anomalies, oc_scores = get_ocsvm_anomalies(models_artifacts["ocsvm"], df_scaled)
                            results_df['is_anomaly_ocsvm'] = oc_anomalies
                            results_df['decision_score_ocsvm'] = oc_scores
                            st.info("Deteksi OC-SVM selesai.", icon="🧩")

                        st.success("Pemrosesan log selesai!", icon="🎉")
                        st.session_state["results_df"] = results_df 
                        st.session_state["last_file_name"] = uploaded_file.name 
                        st.session_state["last_unique_id"] = unique_id 

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}", icon="🔥")
                        print(f"Error processing file: {e}") # Untuk logging di server
                    finally:
                        # Hapus file temporer setelah diproses
                        if os.path.exists(temp_input_filepath):
                            try:
                                os.remove(temp_input_filepath)
                            except Exception as e_del:
                                print(f"Gagal menghapus file temporer {temp_input_filepath}: {e_del}")
                
                # Rerun setelah proses agar hasil langsung ditampilkan di bawah
                # atau update bagian hasil secara langsung jika tidak ingin rerun seluruh halaman
                st.rerun()

    # Tampilkan hasil jika ada di session state
    if "results_df" in st.session_state and st.session_state["results_df"] is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi")
        
        # Filter untuk menampilkan hanya anomali
        show_only_anomalies = st.checkbox("Tampilkan hanya anomali yang terdeteksi", key="show_anomalies_filter")
        
        df_to_display = st.session_state["results_df"]
        if show_only_anomalies:
            anomaly_columns = [col for col in ['is_anomaly_ae', 'is_anomaly_ocsvm'] if col in df_to_display.columns]
            if anomaly_columns:
                # Tampilkan baris jika salah satu model mendeteksi sebagai anomali
                df_to_display = df_to_display[df_to_display[anomaly_columns].any(axis=1)]
            else:
                st.caption("Tidak ada kolom deteksi anomali untuk filter.")

        if df_to_display.empty and show_only_anomalies:
            st.info("Tidak ada anomali yang terdeteksi berdasarkan filter.", icon="ℹ️")
        elif df_to_display.empty:
             st.info("Hasil kosong.", icon="ℹ️")
        else:
            st.dataframe(df_to_display, use_container_width=True, height=400)

            csv_data = convert_df_to_csv(st.session_state["results_df"]) # Selalu download semua hasil
            
            last_name = st.session_state.get("last_file_name", "log")
            last_id = st.session_state.get("last_unique_id", "hasil")
            output_csv_filename = f"hasil_deteksi_{last_id}_{os.path.splitext(last_name)[0]}.csv"
            
            st.download_button(
                label="Unduh Semua Hasil (.csv)",
                data=csv_data,
                file_name=output_csv_filename,
                mime="text/csv",
                use_container_width=True,
                key="download_button_dashboard"
            )
    elif uploaded_file is None:
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")


# Panggil fungsi utama untuk halaman ini
# Ini tidak akan dipanggil jika file ini dijalankan sebagai bagian dari aplikasi multi-halaman Streamlit
# Kecuali jika Anda menjalankan file ini secara langsung (python pages/1_Dashboard.py)
if __name__ == "__main__":
    # Untuk pengujian langsung, pastikan session state disimulasikan
    if "logged_in" not in st.session_state: 
        st.session_state.logged_in = True 
        st.session_state.username = "Test User"
    run_dashboard_page()