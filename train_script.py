# train_script.py
# Skrip untuk melatih model Autoencoder dan One-Class SVM secara lokal.

import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# Impor fungsi-fungsi yang diperlukan dari models.py
try:
    from models import (
        parse_log_file, 
        preprocess_data, 
        train_autoencoder, 
        train_ocsvm
    )
except ImportError:
    print("Error: Pastikan file 'models.py' berada di direktori yang sama dengan skrip ini.")
    exit()

print("="*50)
print("       SKRIP PELATIHAN MODEL DETEKSI ANOMALI       ")
print("="*50)

# --- 1. Konfigurasi Path ---
# GANTI INI DENGAN PATH KE FILE LOG TRAINING NORMAL ANDA!
TRAINING_LOG_FILE_PATH = 'path_ke_file_log_normal_anda.txt' 

# Direktori untuk menyimpan model dan artefak (sama seperti di app.py/streamlit_app.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_MODEL_DIR = os.path.join(BASE_DIR, "trained_models_artifacts")
os.makedirs(BASE_MODEL_DIR, exist_ok=True) # Buat direktori jika belum ada

# Path output
AUTOENCODER_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "trained_autoencoder_model.h5")
OCSVM_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "trained_ocsvm_model.joblib")
SCALER_PATH = os.path.join(BASE_MODEL_DIR, "trained_scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(BASE_MODEL_DIR, "trained_label_encoders.joblib")
TRAINING_MSE_AE_PATH = os.path.join(BASE_MODEL_DIR, "training_mse_ae.npy")

print(f"\n[INFO] Data training akan dibaca dari: {TRAINING_LOG_FILE_PATH}")
print(f"[INFO] Model & Artefak akan disimpan di: {BASE_MODEL_DIR}\n")

# --- 2. Cek File Training ---
if not os.path.exists(TRAINING_LOG_FILE_PATH):
    print(f"[ERROR] File training log '{TRAINING_LOG_FILE_PATH}' tidak ditemukan!")
    print("[ERROR] Silakan perbarui variabel 'TRAINING_LOG_FILE_PATH' di skrip ini dan coba lagi.")
    exit()

# --- 3. Parsing & Pra-pemrosesan Data ---
print("[LANGKAH 1/5] Memulai parsing dan pra-pemrosesan data training...")
start_time = time.time()
df_train_raw = parse_log_file(TRAINING_LOG_FILE_PATH)

if df_train_raw.empty:
    print("[ERROR] Parsing data gagal atau file log kosong.")
    exit()

print(f"[INFO] Parsing selesai. Jumlah record mentah: {len(df_train_raw)}")
print("[INFO] Melakukan pra-pemrosesan (normalisasi, encoding, dll.)...")
df_train_scaled, scaler, label_encoders, feature_cols, _ = preprocess_data(df_train_raw.copy(), is_training=True)

if df_train_scaled is None or df_train_scaled.empty:
    print("[ERROR] Pra-pemrosesan data gagal.")
    exit()

end_time = time.time()
print(f"[SUKSES] Pra-pemrosesan data selesai. Shape data: {df_train_scaled.shape}. Waktu: {end_time - start_time:.2f} detik.")

# --- 4. Simpan Scaler & Label Encoders ---
print("\n[LANGKAH 2/5] Menyimpan Scaler dan Label Encoders...")
try:
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    print(f"[SUKSES] Scaler disimpan di: {SCALER_PATH}")
    print(f"[SUKSES] Label Encoders disimpan di: {LABEL_ENCODERS_PATH}")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan scaler/encoders: {e}")
    exit()

# --- 5. Latih & Simpan Autoencoder ---
print("\n[LANGKAH 3/5] Melatih model Autoencoder...")
start_time = time.time()
input_dim = df_train_scaled.shape[1]
autoencoder_model, history_ae = train_autoencoder(df_train_scaled, input_dim=input_dim, model_save_path=AUTOENCODER_MODEL_PATH, epochs=50) # Anda bisa sesuaikan epochs

if autoencoder_model:
    end_time = time.time()
    print(f"[SUKSES] Model Autoencoder dilatih dan disimpan. Waktu: {end_time - start_time:.2f} detik.")
    
    # --- 6. Hitung & Simpan MSE Training Autoencoder ---
    print("\n[LANGKAH 4/5] Menghitung & Menyimpan MSE Training Autoencoder...")
    try:
        train_predictions_ae = autoencoder_model.predict(df_train_scaled)
        training_mse_ae = np.mean(np.power(df_train_scaled.to_numpy() - train_predictions_ae, 2), axis=1)
        np.save(TRAINING_MSE_AE_PATH, training_mse_ae)
        print(f"[SUKSES] Training MSE untuk Autoencoder disimpan di: {TRAINING_MSE_AE_PATH}")
    except Exception as e:
        print(f"[ERROR] Gagal menghitung atau menyimpan MSE training: {e}")
else:
    print("[ERROR] Pelatihan Autoencoder gagal.")
    exit()

# --- 7. Latih & Simpan One-Class SVM ---
print("\n[LANGKAH 5/5] Melatih model One-Class SVM...")
start_time = time.time()
ocsvm_model = train_ocsvm(df_train_scaled, model_save_path=OCSVM_MODEL_PATH)

if ocsvm_model:
    end_time = time.time()
    print(f"[SUKSES] Model One-Class SVM dilatih dan disimpan. Waktu: {end_time - start_time:.2f} detik.")
else:
    print("[ERROR] Pelatihan One-Class SVM gagal.")
    exit()

print("\n" + "="*50)
print("     PELATIHAN MODEL SELESAI DENGAN SUKSES!     ")
print("="*50)
print("Artefak berikut telah dibuat/diperbarui di folder 'trained_models_artifacts':")
print(f"- {os.path.basename(AUTOENCODER_MODEL_PATH)}")
print(f"- {os.path.basename(OCSVM_MODEL_PATH)}")
print(f"- {os.path.basename(SCALER_PATH)}")
print(f"- {os.path.basename(LABEL_ENCODERS_PATH)}")
print(f"- {os.path.basename(TRAINING_MSE_AE_PATH)}")
print("\nAnda sekarang siap untuk men-deploy aplikasi Streamlit Anda!")
