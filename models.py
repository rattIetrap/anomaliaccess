# models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import re # Untuk parsing log
import joblib # Untuk menyimpan dan memuat model scikit-learn

# --- Fungsi Parsing Log (Sudah Dimodifikasi) ---
def parse_log_file(file_path):
    """
    Mem-parsing file log, secara spesifik mencari dan memproses log
    Fortigate dengan format key=value dan mengabaikan baris log lain yang tidak sesuai.
    """
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_content in f:
                raw_line_for_record = line_content.strip()
                if not raw_line_for_record:
                    continue

                # --- LANGKAH FILTER BARU ---
                # Hanya proses baris yang terlihat seperti log traffic Fortigate.
                # Kita mencari keyword kunci yang harus ada untuk memastikan formatnya benar.
                # Ini membuat parser lebih tangguh terhadap data kotor dari berbagai sumber.
                if 'devid=' not in raw_line_for_record or 'type="traffic"' not in raw_line_for_record or 'logid=' not in raw_line_for_record:
                    # Jika tidak ada keyword ini, anggap bukan log Fortigate traffic yang kita inginkan dan lewati.
                    # print(f"DEBUG: Skipping line (not Fortigate traffic): {raw_line_for_record[:100]}...") # Bisa di-uncomment untuk debug
                    continue
                # --- AKHIR LANGKAH FILTER ---

                # Lanjutkan dengan parsing jika lolos filter
                kv_part_start_index = raw_line_for_record.find("date=") 
                if kv_part_start_index == -1: 
                    kv_part_start_index = raw_line_for_record.find("devname=") 
                
                if kv_part_start_index != -1:
                    kv_string_to_parse = raw_line_for_record[kv_part_start_index:]
                else:
                    kv_string_to_parse = raw_line_for_record

                pairs = re.findall(r'(\w+)=(".*?"|\S+)', kv_string_to_parse)
                record = {key: value.strip('"') for key, value in pairs}
                
                if record: 
                    record['_raw_log_line_'] = raw_line_for_record 
                    records.append(record)
    except FileNotFoundError:
        print(f"Error: File log tidak ditemukan di {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error saat membaca file log {file_path}: {e}")
        return pd.DataFrame()
    
    if not records:
        # Pesan ini sekarang akan muncul jika tidak ada log format Fortigate yang valid ditemukan
        print(f"Peringatan: Tidak ada data log dengan format Fortigate traffic yang valid yang berhasil diparsing dari {file_path}.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df

# --- Fungsi Model (Tetap Sama Seperti Sebelumnya) ---
def create_autoencoder(input_dim, encoding_dim_ratio=0.5, intermediate_dim_ratio=0.75, dropout_rate=0.2):
    # ... (Isi fungsi ini tetap sama) ...
    intermediate_nodes = max(8, int(input_dim * intermediate_dim_ratio)); encoding_nodes = max(4, int(input_dim * encoding_dim_ratio))
    if encoding_nodes >= intermediate_nodes : encoding_nodes = max(4, int(intermediate_nodes / 2))
    input_layer = Input(shape=(input_dim,)); encoder = Dense(intermediate_nodes, activation="relu")(input_layer); encoder = Dropout(dropout_rate)(encoder); encoder = Dense(encoding_nodes, activation="relu")(encoder); decoder = Dense(intermediate_nodes, activation="relu")(encoder); decoder = Dropout(dropout_rate)(decoder); decoder_output = Dense(input_dim, activation="sigmoid")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder_output); autoencoder.compile(optimizer='adam', loss='mse'); return autoencoder

def get_autoencoder_anomalies(autoencoder_model, data_scaled, threshold_percentile=90, training_mse=None):
    # ... (Isi fungsi ini tetap sama) ...
    if data_scaled.empty: return pd.Series(dtype='bool'), pd.Series(dtype='float')
    predictions = autoencoder_model.predict(data_scaled, verbose=0); data_scaled_np = data_scaled.to_numpy() if isinstance(data_scaled, pd.DataFrame) else data_scaled
    mse = np.mean(np.power(data_scaled_np - predictions, 2), axis=1)
    threshold = 0 
    if training_mse is not None and len(training_mse) > 0:
        threshold = np.percentile(training_mse, threshold_percentile)
    else:
        if len(mse) == 0: return pd.Series([False] * len(data_scaled), index=data_scaled.index if hasattr(data_scaled, 'index') else None), pd.Series(dtype='float', index=data_scaled.index if hasattr(data_scaled, 'index') else None)
        threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold
    return pd.Series(anomalies, index=data_scaled.index if hasattr(data_scaled, 'index') else None), pd.Series(mse, index=data_scaled.index if hasattr(data_scaled, 'index') else None)

def create_ocsvm(kernel='rbf', nu=0.05, gamma='auto'):
    # ... (Isi fungsi ini tetap sama) ...
    return OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

def get_ocsvm_anomalies(ocsvm_model, data_scaled):
    # ... (Isi fungsi ini tetap sama) ...
    if data_scaled.empty: return pd.Series(dtype='bool'), pd.Series(dtype='float')
    try:
        predictions = ocsvm_model.predict(data_scaled); decision_scores = ocsvm_model.decision_function(data_scaled)
    except Exception as e:
        print(f"Error saat prediksi OC-SVM: {e}"); idx = data_scaled.index if hasattr(data_scaled, 'index') else None
        return pd.Series([False] * len(data_scaled), index=idx), pd.Series(dtype='float', index=idx)
    anomalies = predictions == -1
    return pd.Series(anomalies, index=data_scaled.index if hasattr(data_scaled, 'index') else None), pd.Series(decision_scores, index=data_scaled.index if hasattr(data_scaled, 'index') else None)

# ip_to_int tidak lagi dipanggil oleh alur kerja utama kita, tapi bisa disimpan jika ada penggunaan lain
def ip_to_int(ip_series):
    return ip_series.apply(lambda ip: int(''.join([f"{int(octet):03d}" for octet in ip.split('.')])) if pd.notnull(ip) and isinstance(ip, str) and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip) else np.nan)
