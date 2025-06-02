# models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # Tidak digunakan langsung di sini jika training hanya di notebook
from sklearn.svm import OneClassSVM
# from sklearn.model_selection import train_test_split # Tidak digunakan di sini jika training hanya di notebook
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Tidak digunakan di sini jika training hanya di notebook
import os
import re # Untuk parsing log
import joblib # Untuk menyimpan dan memuat model scikit-learn

# --- Fungsi Parsing Log ---
def parse_log_file(file_path):
    """
    Mem-parsing file log Fortigate (.txt) menjadi Pandas DataFrame.
    Menangani header syslog dan menyimpan baris log mentah.
    """
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_content in f:
                raw_line_for_record = line_content.strip()
                if not raw_line_for_record: # Lewati baris kosong
                    continue
                
                # Cari awal dari bagian key=value (misalnya, dimulai dengan "date=")
                kv_part_start_index = raw_line_for_record.find("date=") 
                if kv_part_start_index == -1: 
                    kv_part_start_index = raw_line_for_record.find("devname=") 
                
                if kv_part_start_index != -1:
                    kv_string_to_parse = raw_line_for_record[kv_part_start_index:]
                else:
                    # Jika tidak ada penanda, coba parse seluruh baris
                    # Ini mungkin perlu penyesuaian jika ada baris non-key-value yang valid
                    # print(f"Peringatan: Penanda 'date=' atau 'devname=' tidak ditemukan pada baris, mencoba parse seluruh baris: {raw_line_for_record[:100]}...")
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
        print(f"Tidak ada data yang berhasil diparsing dari {file_path}.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df

# --- Fungsi Model Autoencoder ---
# Definisi create_autoencoder bisa tetap di sini jika dipanggil dari notebook atau skrip lain
# Namun, training utama ada di notebook.
def create_autoencoder(input_dim, encoding_dim_ratio=0.5, intermediate_dim_ratio=0.75, dropout_rate=0.2):
    """
    Membuat model Autoencoder dengan arsitektur yang lebih fleksibel.
    encoding_dim_ratio: rasio untuk layer bottleneck (misal 0.5 dari input_dim).
    intermediate_dim_ratio: rasio untuk layer intermediate (misal 0.75 dari input_dim).
    """
    intermediate_nodes = max(8, int(input_dim * intermediate_dim_ratio))
    encoding_nodes = max(4, int(input_dim * encoding_dim_ratio))
    
    if encoding_nodes >= intermediate_nodes : # Pastikan bottleneck lebih kecil
        encoding_nodes = max(4, int(intermediate_nodes / 2))


    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(intermediate_nodes, activation="relu")(input_layer)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(encoding_nodes, activation="relu")(encoder) # Bottleneck
    
    # Decoder
    decoder = Dense(intermediate_nodes, activation="relu")(encoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder_output = Dense(input_dim, activation="sigmoid")(decoder) 
    
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Fungsi train_autoencoder di sini mungkin tidak lagi digunakan jika semua training di notebook
# Namun, bisa dipertahankan jika ada skenario penggunaan lain.
# Jika hanya notebook, fungsi ini bisa dihapus dari models.py untuk menyederhanakan.
# Untuk saat ini, saya akan membiarkannya dengan catatan.

# def train_autoencoder(X_train, input_dim, model_save_path="autoencoder_model.h5", epochs=50, batch_size=32):
#     """Melatih model Autoencoder dan menyimpannya."""
#     # ... (Implementasi dari models.py asli Anda bisa diletakkan di sini jika masih relevan) ...
#     # ... (Namun, train_script.ipynb sudah memiliki logika training sendiri) ...
#     print("Fungsi train_autoencoder di models.py tidak lagi menjadi metode training utama. Gunakan train_script.ipynb.")
#     return None, None


def get_autoencoder_anomalies(autoencoder_model, data_scaled, threshold_percentile=95, training_mse=None):
    """
    Mendeteksi anomali menggunakan Autoencoder.
    Jika training_mse disediakan, threshold dihitung dari situ. Jika tidak, dari data_scaled.
    """
    if data_scaled.empty:
        return pd.Series(dtype='bool'), pd.Series(dtype='float')

    predictions = autoencoder_model.predict(data_scaled, verbose=0) # verbose=0 untuk mematikan log predict
    data_scaled_np = data_scaled.to_numpy() if isinstance(data_scaled, pd.DataFrame) else data_scaled
    
    mse = np.mean(np.power(data_scaled_np - predictions, 2), axis=1)
    
    threshold = 0 # Inisialisasi threshold
    if training_mse is not None and len(training_mse) > 0:
        threshold = np.percentile(training_mse, threshold_percentile)
    else:
        if len(mse) == 0: 
            # print("Peringatan: MSE dari data input kosong, tidak bisa menghitung threshold dinamis.") # Sudah ditangani di dashboard
            # Kembalikan semua sebagai non-anomali jika tidak ada MSE untuk dihitung
            return pd.Series([False] * len(data_scaled), index=data_scaled.index if hasattr(data_scaled, 'index') else None), \
                   pd.Series(dtype='float', index=data_scaled.index if hasattr(data_scaled, 'index') else None)
        threshold = np.percentile(mse, threshold_percentile) 
        # Pesan warning sudah ada di dashboard jika training_mse tidak ada
        # print(f"Peringatan: Threshold Autoencoder dihitung dari data saat ini (data_scaled): {threshold:.6f}. Idealnya dari training_mse.")
    
    anomalies = mse > threshold
    return pd.Series(anomalies, index=data_scaled.index if hasattr(data_scaled, 'index') else None), \
           pd.Series(mse, index=data_scaled.index if hasattr(data_scaled, 'index') else None)

# --- Fungsi Model One-Class SVM ---
# Definisi create_ocsvm bisa tetap di sini
def create_ocsvm(kernel='rbf', nu=0.05, gamma='auto'): # gamma='auto' atau 'scale'
    """Membuat model One-Class SVM."""
    ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    return ocsvm

# Fungsi train_ocsvm di sini mungkin tidak lagi digunakan jika semua training di notebook
# def train_ocsvm(X_train, model_save_path="ocsvm_model.joblib"):
#     """Melatih model One-Class SVM dan menyimpannya."""
#     # ... (Implementasi dari models.py asli Anda bisa diletakkan di sini jika masih relevan) ...
#     # ... (Namun, train_script.ipynb sudah memiliki logika training sendiri) ...
#     print("Fungsi train_ocsvm di models.py tidak lagi menjadi metode training utama. Gunakan train_script.ipynb.")
#     return None

def get_ocsvm_anomalies(ocsvm_model, data_scaled):
    """Mendeteksi anomali menggunakan One-Class SVM."""
    if data_scaled.empty:
         return pd.Series(dtype='bool'), pd.Series(dtype='float')
    try:
        predictions = ocsvm_model.predict(data_scaled) 
        decision_scores = ocsvm_model.decision_function(data_scaled)
    except Exception as e:
        print(f"Error saat prediksi OC-SVM: {e}")
        # Kembalikan series kosong atau sesuai dengan panjang input jika error
        idx = data_scaled.index if hasattr(data_scaled, 'index') else None
        return pd.Series([False] * len(data_scaled), index=idx), pd.Series(dtype='float', index=idx)

    anomalies = predictions == -1 
    return pd.Series(anomalies, index=data_scaled.index if hasattr(data_scaled, 'index') else None), \
           pd.Series(decision_scores, index=data_scaled.index if hasattr(data_scaled, 'index') else None)


# Fungsi ip_to_int dan preprocess_data LAMA Anda dari models.py asli
# Kemungkinan besar TIDAK LAGI DIGUNAKAN secara langsung oleh dashboard
# jika pra-pemrosesan utama dilakukan di train_script.ipynb dan
# fungsi preprocess_dashboard_data (di 1_Dashboard.py) menangani data baru.
# Saya akan tetap menyertakannya di sini jika ada bagian lain yang mungkin masih merujuknya,
# namun tandai sebagai potensial untuk dihapus atau direfaktor.

# def ip_to_int_lama(ip_series):
#     """Mengonversi serangkaian alamat IP string menjadi integer. (VERSI LAMA)"""
#     return ip_series.apply(lambda ip: int(''.join([f"{int(octet):03d}" for octet in ip.split('.')])) if pd.notnull(ip) and isinstance(ip, str) and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip) else np.nan)

# def preprocess_data_lama(df, scaler=None, label_encoders=None, is_training=True):
#     """
#     Melakukan pra-pemrosesan pada DataFrame log. (VERSI LAMA DARI models.py ASLI ANDA)
#     FUNGSI INI KEMUNGKINAN TIDAK SESUAI LAGI DENGAN ARTEFAK DARI train_script.ipynb BARU.
#     """
#     print("PERINGATAN: Fungsi preprocess_data_lama sedang dipanggil. Pastikan ini sesuai dengan alur kerja Anda.")
#     if df.empty:
#         return pd.DataFrame(), None, None, None, None
#     # ... (Implementasi fungsi preprocess_data lama Anda dari models.py asli) ...
#     # ... (Perlu direview dan disesuaikan atau dihapus jika tidak relevan lagi) ...
#     # Contoh:
#     # required_cols = ['date', 'time', 'srcip', 'srccountry', 'dstip', 'dstport', 'action']
#     # ... (dst)
#     # return df_scaled, scaler, label_encoders, feature_columns, df_original_for_output
#     pass # Hapus pass dan isi dengan implementasi lama jika perlu dipertahankan dan direview