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

# --- Fungsi Pra-pemrosesan Data ---
def ip_to_int(ip_series):
    """Mengonversi serangkaian alamat IP string menjadi integer."""return ip_series.apply(lambda ip: int(''.join([f"{int(octet):03d}" for octet in ip.split('.')])) if pd.notnull(ip) and isinstance(ip, str) and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip) else np.nan)

def parse_log_file(file_path):
    """
    Mem-parsing file log Fortigate (.txt) menjadi Pandas DataFrame.
    Format log diasumsikan sebagai baris-baris key=value.
    SETIAP RECORD SEKARANG AKAN MENYERTAKAN FIELD '_raw_log_line_'
    """
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_content in f: # Tidak perlu enumerate jika tidak dipakai
                raw_line_for_record = line_content.strip() # Simpan baris log mentah
                if not raw_line_for_record: # Lewati baris kosong
                    continue
                
                # Mencocokkan pasangan key=value, termasuk value yang mengandung spasi jika diapit tanda kutip
                pairs = re.findall(r'(\w+)=(".*?"|\S+)', line_content)
                record = {key: value.strip('"') for key, value in pairs}
                
                if record: # Hanya tambahkan jika record tidak kosong
                    record['_raw_log_line_'] = raw_line_for_record # Tambahkan baris log mentah ke record
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

def preprocess_data(df, scaler=None, label_encoders=None, is_training=True):
    """
    Melakukan pra-pemrosesan pada DataFrame log.
    - Memilih fitur
    - Menangani nilai yang hilang
    - Mengonversi tipe data
    - Melakukan encoding dan normalisasi
    - Membuat fitur baru (freq_access)
    Jika is_training=False, scaler dan label_encoders harus disediakan.
    """
    if df.empty:
        return pd.DataFrame(), None, None, None, None

    required_cols = ['date', 'time', 'srcip', 'srccountry', 'dstip', 'dstport', 'action']
    # Cek kolom esensial untuk proses awal
    essential_initial_cols = ['date', 'time', 'srcip', 'dstport', 'action']
    if not all(col in df.columns for col in essential_initial_cols):
        missing_essentials = [col for col in essential_initial_cols if col not in df.columns]
        print(f"Kolom penting awal {missing_essentials} tidak ditemukan dalam DataFrame input.")
        return pd.DataFrame(), scaler, label_encoders, None, None

    # Pilih kolom yang tersedia dari daftar required_cols
    available_cols = [col for col in required_cols if col in df.columns]
    df_processed = df[available_cols].copy()

    # Konversi Timestamp
    try:
        df_processed['timestamp_dt'] = pd.to_datetime(df_processed['date'] + ' ' + df_processed['time'], errors='coerce')
        df_processed['timestamp'] = (df_processed['timestamp_dt'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df_processed.drop(columns=['date', 'time', 'timestamp_dt'], inplace=True, errors='ignore')
    except Exception as e:
        print(f"Error saat konversi timestamp: {e}")
        df_processed['timestamp'] = np.nan

    # Konversi dstport ke numerik
    df_processed['dstport'] = pd.to_numeric(df_processed['dstport'], errors='coerce')

    # Penanganan Nilai Hilang pada kolom kunci
    key_cols_for_na_check = ['srcip', 'dstport', 'timestamp', 'action']
    # Tambahkan kolom yang mungkin tidak ada dari available_cols tapi ada di key_cols_for_na_check
    for col in key_cols_for_na_check:
        if col not in df_processed.columns:
            df_processed[col] = np.nan
            
    df_processed.dropna(subset=[col for col in key_cols_for_na_check if col in df_processed.columns], inplace=True)
    if df_processed.empty:
        print("DataFrame kosong setelah menangani nilai hilang pada kolom kunci.")
        return pd.DataFrame(), scaler, label_encoders, None, None

    # Isi NaN yang tersisa
    if 'srccountry' in df_processed.columns:
        df_processed['srccountry'].fillna('Unknown', inplace=True)
    else:
        df_processed['srccountry'] = 'Unknown'
    
    if 'dstip' in df_processed.columns:
         df_processed['dstip'].fillna('0.0.0.0', inplace=True)
    else:
        df_processed['dstip'] = '0.0.0.0'

    # Transformasi IP ke Integer
    df_processed['srcip_int'] = ip_to_int(df_processed['srcip'])
    df_processed.dropna(subset=['srcip_int'], inplace=True)
    if df_processed.empty:
        print("DataFrame kosong setelah konversi srcip ke int dan dropna.")
        return pd.DataFrame(), scaler, label_encoders, None, None

    # Label Encoding
    if is_training:
        label_encoders = {}
        if 'srccountry' in df_processed.columns:
            le_srccountry = LabelEncoder()
            # Pastikan semua nilai adalah string sebelum fit_transform
            df_processed['srccountry_enc'] = le_srccountry.fit_transform(df_processed['srccountry'].astype(str))
            label_encoders['srccountry'] = le_srccountry
        else: # Jika kolom srccountry tidak ada sama sekali di input
            df_processed['srccountry_enc'] = 0 
    else: # Saat prediksi/testing
        if 'srccountry' in df_processed.columns and label_encoders and 'srccountry' in label_encoders:
            le_srccountry = label_encoders['srccountry']
            # Tangani label baru yang tidak ada saat training
            df_processed['srccountry_enc'] = df_processed['srccountry'].astype(str).apply(
                lambda x: le_srccountry.transform([x])[0] if x in le_srccountry.classes_ else -1 # -1 untuk unknown
            )
            # Jika ada -1, bisa diisi dengan nilai modus dari data training atau nilai khusus
            if -1 in df_processed['srccountry_enc'].unique():
                 print("Peringatan: Ditemukan srccountry baru saat prediksi. Ditandai sebagai -1.")
        else:
            df_processed['srccountry_enc'] = 0 


    action_mapping = {'accept': 0, 'deny': 1, 'close': 2, 'timeout': 3} # Sesuai skripsi [cite: 7377]
    # Pastikan kolom 'action' ada dan merupakan string sebelum mapping
    if 'action' in df_processed.columns:
        df_processed['action_enc'] = df_processed['action'].astype(str).map(action_mapping).fillna(max(action_mapping.values()) + 1) # Isi NaN dengan nilai baru
    else:
        df_processed['action_enc'] = 0 # Default jika tidak ada


    # Fitur Frekuensi Akses IP
    if 'srcip' in df_processed.columns:
        srcip_value_counts = df_processed['srcip'].value_counts(normalize=True)
        df_processed['freq_access'] = df_processed['srcip'].map(srcip_value_counts).fillna(0)
    else:
        df_processed['freq_access'] = 0.0


    feature_columns = ['srcip_int', 'srccountry_enc', 'dstport', 'action_enc', 'timestamp', 'freq_access']
    
    # Pastikan semua feature_columns ada, jika tidak, tambahkan dengan nilai default
    for col in feature_columns:
        if col not in df_processed.columns:
            print(f"Peringatan: Kolom fitur '{col}' tidak ditemukan. Menggunakan nilai default 0.")
            df_processed[col] = 0 

    df_model_input = df_processed[feature_columns].copy()
    df_model_input.fillna(0, inplace=True) # Isi NaN yang mungkin masih ada dengan 0 sebelum scaling

    # Normalisasi Fitur Numerik
    if is_training:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_model_input)
    else:
        if scaler is None:
            print("Error: Scaler harus disediakan saat is_training=False.")
            return pd.DataFrame(), scaler, label_encoders, None, None
        # Cek apakah jumlah fitur sama
        if df_model_input.shape[1] != scaler.n_features_in_:
            print(f"Error: Jumlah fitur input ({df_model_input.shape[1]}) tidak cocok dengan scaler ({scaler.n_features_in_}).")
            return pd.DataFrame(), scaler, label_encoders, None, None
        scaled_data = scaler.transform(df_model_input)

    df_scaled = pd.DataFrame(scaled_data, columns=feature_columns, index=df_model_input.index)
    
    original_cols_for_output = ['srcip', 'srccountry', 'dstip', 'dstport', 'action', 'timestamp']
    original_cols_for_output = [col for col in original_cols_for_output if col in df_processed.columns]
    # Pastikan index sinkron dan kolom ada sebelum slicing
    valid_indices = df_scaled.index.intersection(df_processed.index)
    df_original_for_output = df_processed.loc[valid_indices, original_cols_for_output].copy()


    return df_scaled, scaler, label_encoders, feature_columns, df_original_for_output


# --- Model Autoencoder ---
def create_autoencoder(input_dim, encoding_dim=32, dropout_rate=0.2): # Arsitektur bisa disesuaikan [cite: 7574, 7576, 7577, 7580]
    """Membuat model Autoencoder."""
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(128, activation="relu")(input_layer)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(encoding_dim, activation="relu")(encoder) # Bottleneck
    
    # Decoder
    decoder = Dense(64, activation="relu")(encoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder_output = Dense(input_dim, activation="sigmoid")(decoder) # Sigmoid karena data dinormalisasi [0,1] [cite: 7577]
    
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error untuk reconstruction loss [cite: 7578]
    return autoencoder

def train_autoencoder(X_train, input_dim, model_save_path="autoencoder_model.h5", epochs=50, batch_size=32):
    """Melatih model Autoencoder dan menyimpannya."""
    if X_train.empty or X_train.shape[0] == 0:
        print("Data training kosong, tidak bisa melatih Autoencoder.")
        return None, None # Kembalikan dua nilai agar unpacking tidak error
        
    autoencoder = create_autoencoder(input_dim=input_dim)
    
    # Cek apakah data cukup untuk split
    if X_train.shape[0] < 10: # Minimal sampel untuk split (contoh)
        print("Data training tidak cukup untuk dibagi menjadi train/validation. Melatih dengan semua data.")
        X_train_data_fit = X_train
        X_val_data_fit = X_train # Gunakan data yang sama jika tidak bisa split
    else:
        X_train_split, X_val_split = train_test_split(X_train, test_size=0.2, random_state=42)
        if X_train_split.empty or X_val_split.empty: # Double check setelah split
            print("Split menghasilkan set kosong, melatih dengan semua data.")
            X_train_data_fit = X_train
            X_val_data_fit = X_train
        else:
            X_train_data_fit = X_train_split
            X_val_data_fit = X_val_split


    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # [cite: 6823]
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=0) 
    
    print(f"Melatih Autoencoder dengan {epochs} epochs...")
    history = autoencoder.fit(X_train_data_fit, X_train_data_fit,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_val_data_fit, X_val_data_fit),
                              callbacks=[early_stopping, model_checkpoint],
                              verbose=1) 
    
    print(f"Autoencoder dilatih dan model terbaik disimpan di {model_save_path}")
    # Muat model terbaik yang disimpan oleh ModelCheckpoint
    try:
        best_model = load_model(model_save_path)
    except Exception as e:
        print(f"Gagal memuat model terbaik dari {model_save_path}: {e}. Mengembalikan model terakhir.")
        best_model = autoencoder 
    return best_model, history

def get_autoencoder_anomalies(autoencoder_model, data_scaled, threshold_percentile=95, training_mse=None): # Threshold dari persentil [cite: 6744, 6962]
    """
    Mendeteksi anomali menggunakan Autoencoder.
    Jika training_mse disediakan, threshold dihitung dari situ. Jika tidak, dari data_scaled.
    """
    if data_scaled.empty:
        return pd.Series(dtype='bool'), pd.Series(dtype='float')

    predictions = autoencoder_model.predict(data_scaled)
    data_scaled_np = data_scaled.to_numpy() if isinstance(data_scaled, pd.DataFrame) else data_scaled
    
    mse = np.mean(np.power(data_scaled_np - predictions, 2), axis=1) # MSE Reconstruction Error [cite: 6959]
    
    if training_mse is not None and len(training_mse) > 0:
        threshold = np.percentile(training_mse, threshold_percentile) # [cite: 6744]
    else:
        if len(mse) == 0: 
            print("Peringatan: MSE kosong, tidak bisa menghitung threshold.")
            return pd.Series(dtype='bool', index=data_scaled.index), pd.Series(dtype='float', index=data_scaled.index)
        threshold = np.percentile(mse, threshold_percentile) 
        print(f"Peringatan: Threshold Autoencoder dihitung dari data saat ini (data_scaled): {threshold:.6f}. Idealnya dari training_mse.")
    
    anomalies = mse > threshold # Data di atas threshold adalah anomali [cite: 6952]
    return pd.Series(anomalies, index=data_scaled.index), pd.Series(mse, index=data_scaled.index)

# --- Model One-Class SVM ---
def create_ocsvm(kernel='rbf', nu=0.05, gamma='scale'): # Parameter OC-SVM [cite: 6737, 6983, 6984]
    """Membuat model One-Class SVM."""
    ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    return ocsvm

def train_ocsvm(X_train, model_save_path="ocsvm_model.joblib"):
    """Melatih model One-Class SVM dan menyimpannya."""
    if X_train.empty:
        print("Data training kosong, tidak bisa melatih OC-SVM.")
        return None
        
    ocsvm = create_ocsvm()
    print("Melatih OC-SVM...")
    ocsvm.fit(X_train) # OC-SVM dilatih hanya dengan data normal [cite: 6978]
    
    joblib.dump(ocsvm, model_save_path)
    print(f"OC-SVM dilatih dan disimpan di {model_save_path}")
    return ocsvm

def get_ocsvm_anomalies(ocsvm_model, data_scaled):
    """Mendeteksi anomali menggunakan One-Class SVM."""
    if data_scaled.empty:
         return pd.Series(dtype='bool'), pd.Series(dtype='float')
    predictions = ocsvm_model.predict(data_scaled) 
    decision_scores = ocsvm_model.decision_function(data_scaled) # Skor keputusan [cite: 6986]
    anomalies = predictions == -1 # Skor negatif (atau -1) adalah anomali [cite: 6987, 7002]
    return pd.Series(anomalies, index=data_scaled.index), pd.Series(decision_scores, index=data_scaled.index)

