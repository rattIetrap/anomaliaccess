# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import time 
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf # ### Pastikan impor ini ada ###

# Impor fungsi dari models.py 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models import parse_log_file, preprocess_data, get_autoencoder_anomalies, get_ocsvm_anomalies
except ImportError as e:
    # Penanganan error impor yang lebih baik
    if 'streamlit_app_run_first' not in st.session_state: # Flag sederhana, bisa diatur di streamlit_app.py
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

# Path ke Model dan Artefak (pastikan nama file konsisten dengan yang disimpan oleh train_script.py)
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
    
    # Fungsi internal untuk memuat dan menangani error
    def check_and_load(path, name, load_func, type_name, icon, custom_objects_dict=None):
        if os.path.exists(path):
            try:
                if custom_objects_dict:
                    models_artifacts[name] = load_func(path, custom_objects=custom_objects_dict)
                else:
                    models_artifacts[name] = load_func(path)
                models_artifacts["messages"].append(("success", f"{type_name} '{os.path.basename(path)}' berhasil dimuat.", icon))
            except Exception as e:
                # Sertakan detail error yang lebih lengkap
                full_error_message = f"Gagal memuat {type_name} '{os.path.basename(path)}': {str(e)}"
                print(full_error_message) # Cetak ke konsol server untuk debugging
                models_artifacts["messages"].append(("error", full_error_message, "🔥"))
                models_artifacts[name] = None
                models_artifacts["loaded_successfully"] = False
        else:
            models_artifacts["messages"].append(("error", f"File {type_name} tidak ditemukan: {os.path.basename(path)}", "🚨"))
            models_artifacts[name] = None
            models_artifacts["loaded_successfully"] = False

    # Memuat model Autoencoder dengan custom_objects
    check_and_load(AUTOENCODER_MODEL_PATH, "autoencoder", load_model, "Model Autoencoder", "🤖", 
                   custom_objects_dict={'mse': tf.keras.losses.MeanSquaredError()}) # <<< PERUBAHAN DI SINI
    
    check_and_load(OCSVM_MODEL_PATH, "ocsvm", joblib.load, "Model OC-SVM", "🧩")
    check_and_load(SCALER_PATH, "scaler", joblib.load, "Scaler", "⚙️")
    check_and_load(LABEL_ENCODERS_PATH, "label_encoders", joblib.load, "Label Encoders", "🏷️")
    
    # Pemuatan training_mse_ae (tidak fatal jika tidak ada, tapi berikan warning)
    if os.path.exists(TRAINING_MSE_AE_PATH):
        try:
            models_artifacts["training_mse_ae"] = np.load(TRAINING_MSE_AE_PATH)
            models_artifacts["messages"].append(("success", f"Training MSE AE '{os.path.basename(TRAINING_MSE_AE_PATH)}' berhasil dimuat.", "📊"))
        except Exception as e:
            models_artifacts["messages"].append(("error", f"Gagal memuat Training MSE AE '{os.path.basename(TRAINING_MSE_AE_PATH)}': {e}", "🔥"))
            models_artifacts["training_mse_ae"] = None # Set None jika gagal
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
    # Konfigurasi halaman sebaiknya hanya di skrip utama (streamlit_app.py)
    # st.set_page_config(page_title="Dashboard Deteksi", layout="wide", initial_sidebar_state="auto")
    
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Anda harus login untuk mengakses halaman ini.")
        st.page_link("streamlit_app.py", label="Kembali ke Halaman Login", icon="🏠")
        st.stop() 

    st.title("🚀 Dashboard Deteksi Anomali Akses Jaringan")
    
    # Tombol logout di sidebar (sudah ada di streamlit_app.py, tapi bisa juga di sini jika diinginkan)
    # st.sidebar.markdown("---") 
    # st.sidebar.write(f"Login sebagai: **{st.session_state.get('username', 'Pengguna')}**") # Sudah ada di app utama
    # if st.sidebar.button("Logout dari Dashboard", key="dashboard_logout_button_unique", use_container_width=True):
    #     for key in list(st.session_state.keys()): 
    #         del st.session_state[key]
    #     st.switch_page("streamlit_app.py") 

    models_artifacts = load_anomaly_models_and_artifacts()
    
    with st.expander("ℹ️ Status Pemuatan Model & Artefak", expanded=not models_artifacts["loaded_successfully"]):
        for type_msg, msg, icon in models_artifacts.get("messages", []):
            if type_msg == "success": st.success(msg, icon=icon)
            elif type_msg == "error": st.error(msg, icon=icon)
            elif type_msg == "warning": st.warning(msg, icon=icon)

    if not models_artifacts["loaded_successfully"]:
        st.error("Satu atau lebih model/artefak penting gagal dimuat. Fungsi deteksi mungkin tidak akan bekerja dengan benar. Pastikan semua artefak ada di folder `trained_models_artifacts` dan skrip `train_script.py` sudah dijalankan dengan sukses.", icon="💔")

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
        # Gunakan nama file asli untuk file temporer agar lebih mudah dikenali (meskipun ada ID unik)
        temp_input_filename = f"{unique_id}_{uploaded_file.name}"
        temp_input_filepath = os.path.join(UPLOAD_FOLDER, temp_input_filename)
        
        with open(temp_input_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.header("2. Opsi Deteksi & Proses")
        
        ae_available = models_artifacts.get("autoencoder") is not None and models_artifacts.get("scaler") is not None
        ocsvm_available = models_artifacts.get("ocsvm") is not None and models_artifacts.get("scaler") is not None

        col1, col2 = st.columns(2)
        with col1:
            run_autoencoder = st.checkbox("Gunakan Model Autoencoder", value=True, key="cb_ae_dashboard_main_unique_key", disabled=not ae_available)
            if not ae_available: st.caption("Model Autoencoder tidak dapat dimuat.")
        with col2:
            run_ocsvm = st.checkbox("Gunakan Model One-Class SVM", value=True, key="cb_ocsvm_dashboard_main_unique_key", disabled=not ocsvm_available)
            if not ocsvm_available: st.caption("Model OC-SVM tidak dapat dimuat.")

        if st.button("Proses Log Sekarang 🔎", type="primary", use_container_width=True, disabled=not models_artifacts["loaded_successfully"]):
            if not run_autoencoder and not run_ocsvm:
                st.warning("Pilih setidaknya satu model deteksi untuk diproses.", icon="⚠️")
            elif (run_autoencoder and not ae_available) or (run_ocsvm and not ocsvm_available):
                st.error("Satu atau lebih model yang Anda pilih tidak tersedia atau gagal dimuat. Periksa status pemuatan model di atas.", icon="🚫")
            else:
                with st.spinner("Sedang memproses log... Ini mungkin memakan waktu beberapa saat, mohon tunggu. ⏳"):
                    process_start_time = time.time()
                    try:
                        df_raw = parse_log_file(temp_input_filepath)
                        if df_raw.empty:
                            st.error("File log yang diunggah kosong atau gagal diparsing.", icon="❌")
                            # Hapus file temp jika ada
                            if os.path.exists(temp_input_filepath): os.remove(temp_input_filepath)
                            # Hentikan eksekusi lebih lanjut di blok ini jika df_raw kosong
                        else:
                            df_scaled, _, _, _, df_original_for_output = preprocess_data(
                                df_raw.copy(), 
                                scaler=models_artifacts.get("scaler"), 
                                label_encoders=models_artifacts.get("label_encoders"), 
                                is_training=False
                            )

                            if df_scaled is None or df_scaled.empty or df_original_for_output is None or df_original_for_output.empty:
                                st.error("Pra-pemrosesan data gagal atau menghasilkan data kosong.", icon="❌")
                            else:
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

                                st.success(f"Pemrosesan log selesai! Waktu: {time.time() - process_start_time:.2f} detik.", icon="🎉")
                                st.session_state["results_df"] = results_df 
                                st.session_state["last_file_name"] = uploaded_file.name 
                                st.session_state["last_unique_id"] = unique_id
                                # Tidak perlu rerun di sini, biarkan hasil ditampilkan di bawah
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
                # Setelah tombol proses ditekan dan selesai, kita ingin hasil ditampilkan
                # st.rerun() bisa digunakan jika ada perubahan state yang perlu direfleksikan segera
                # Namun, karena hasil disimpan di session_state, bagian di bawah akan otomatis update

    # Tampilkan hasil jika ada di session state
    if "results_df" in st.session_state and st.session_state["results_df"] is not None:
        st.markdown("---")
        st.header("3. Hasil Deteksi")
        
        results_to_show = st.session_state["results_df"]
        
        # Opsi filter tambahan
        st.markdown("**Filter Tampilan Hasil:**")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            show_only_anomalies_ae = st.checkbox("Hanya anomali Autoencoder", key="filter_ae_main_unique", value=False)
        with col_filter2:
            show_only_anomalies_ocsvm = st.checkbox("Hanya anomali OC-SVM", key="filter_ocsvm_main_unique", value=False)

        if show_only_anomalies_ae and 'is_anomaly_ae' in results_to_show.columns:
            results_to_show = results_to_show[results_to_show['is_anomaly_ae'] == True]
        
        if show_only_anomalies_ocsvm and 'is_anomaly_ocsvm' in results_to_show.columns:
            # Jika filter AE juga aktif, filter dari hasil yang sudah difilter AE
            # Jika tidak, filter dari hasil original
            results_to_show = results_to_show[results_to_show['is_anomaly_ocsvm'] == True]
            
        if results_to_show.empty:
            st.info("Tidak ada data yang sesuai dengan filter yang dipilih.", icon="ℹ️")
        else:
            st.dataframe(results_to_show, use_container_width=True, height=400)

        # Tombol download selalu untuk semua hasil (sebelum difilter untuk tampilan)
        csv_data = convert_df_to_csv(st.session_state["results_df"]) 
        
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
    elif uploaded_file is None and models_artifacts.get("loaded_successfully", False): # Periksa apakah model berhasil dimuat
        st.info("Silakan unggah file log untuk memulai analisis.", icon="📤")

# Panggil fungsi utama untuk halaman ini
if __name__ == "__main__":
    # Ini akan dijalankan jika Anda menjalankan `python pages/1_Dashboard.py` secara langsung
    # Untuk pengujian, pastikan st.session_state disimulasikan
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True # Simulasikan login untuk pengujian langsung
        st.session_state.username = "Penguji Dashboard"
        # st.session_state.streamlit_app_run_first = True # Tandai bahwa dashboard dijalankan
    
    run_dashboard_page()
