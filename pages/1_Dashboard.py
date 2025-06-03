# train_script.ipynb

# Sel 1: Impor Pustaka
# =============================================================================
# ### 1. Impor Pustaka 📚
# =============================================================================
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore') # Sebaiknya hanya untuk development
print("✅ Pustaka berhasil diimpor.")

# Sel 2: Memuat Data Log Normal untuk Training
# =============================================================================
# ### 2. Memuat Data Log Normal untuk Training 📂
# =============================================================================
# !!! GANTI INI DENGAN PATH KE FILE DATA TRAINING NORMAL ANDA !!!
# File ini harus berisi log yang Anda anggap 100% normal.
# Formatnya diasumsikan memiliki header syslog diikuti key=value.
file_path_normal_training = 'data/syslog_normal_tgl_01.txt' # Pastikan path ini benar

# Menggunakan fungsi parse_log_file dari models.py
# Pastikan models.py ada di path Python atau di direktori yang sama dengan notebook ini
try:
    # Hapus sys.path append jika models.py ada di direktori yang sama atau sudah di PYTHONPATH
    # import sys
    # # Asumsi notebook ada di root proyek, dan models.py juga di root
    # if '.' not in sys.path: # Tambahkan direktori saat ini ke path jika belum ada
    # sys.path.append('.')
    from models import parse_log_file
except ImportError:
    print("❌ Error: Tidak dapat mengimpor parse_log_file dari models.py.")
    print("Pastikan models.py dapat diakses (misalnya, di direktori yang sama atau di PYTHONPATH).")
    raise

print(f"Memuat data log normal untuk training dari: {file_path_normal_training}")
if not os.path.exists(file_path_normal_training):
    print(f"❌ ERROR: File training '{file_path_normal_training}' TIDAK DITEMUKAN.")
    raise FileNotFoundError(f"File training tidak ditemukan: {file_path_normal_training}")

df_logs = parse_log_file(file_path_normal_training) #

if df_logs.empty:
    print(f"❌ Error: Tidak ada data yang berhasil diparsing dari {file_path_normal_training} atau file kosong.")
    raise ValueError("DataFrame df_logs kosong setelah parsing. Training tidak dapat dilanjutkan.")
else:
    print(f"✅ Berhasil memuat dan mem-parsing {len(df_logs)} baris log normal untuk training.")
    print("\n📋 Pratinjau Data Log Normal (5 baris pertama):")
    print(df_logs.head())
    if '_raw_log_line_' in df_logs.columns:
        print("\n📋 Contoh _raw_log_line_ (menunjukkan baris log asli termasuk header syslog):")
        for line in df_logs['_raw_log_line_'].head(2).to_list():
            print(line)

# Sel 3: Informasi Awal df_logs (Opsional)
# =============================================================================
# ### 3. Info Awal df_logs (Opsional)
# =============================================================================
if not df_logs.empty:
    print("\nℹ️ Informasi DataFrame Training Awal (df_logs):")
    df_logs.info()
    
    # Opsional: Cek dan hapus duplikat jika ada di data training Anda
    # initial_rows = len(df_logs)
    # if df_logs.duplicated().sum() > 0:
    #     print(f"\nJumlah duplikat dalam df_logs: {df_logs.duplicated().sum()}")
    #     df_logs.drop_duplicates(inplace=True)
    #     df_logs.reset_index(drop=True, inplace=True)
    #     print(f"Jumlah data setelah drop duplikat: {len(df_logs)} (dari {initial_rows})")
    # else:
    #     print("\nTidak ada baris duplikat yang ditemukan di df_logs.")
else:
    print("⚠️ df_logs kosong, tidak ada info untuk ditampilkan.")


# Sel 4: Pra-pemrosesan Data Training (Feature Engineering & Scaling)
# =============================================================================
# ### 4. Pra-pemrosesan Data Training (Feature Engineering & Scaling) ⚙️
# =============================================================================

if df_logs.empty:
    print("❌ DataFrame 'df_logs' kosong. Langkah pra-pemrosesan tidak dapat dilanjutkan.")
    # Definisikan variabel agar sel berikutnya tidak langsung error, meskipun training akan gagal
    df_processed = pd.DataFrame()
    label_encoders = {}
    scaler = None
    available_categorical_original = []
    available_numerical_original = []
    df_combined_processed_cols = []
else:
    print(f"\nMemulai pra-pemrosesan untuk {len(df_logs)} baris log normal training...")

    # === 1. DEFINISIKAN FITUR YANG AKAN DIGUNAKAN ===
    # Berdasarkan diskusi dan EDA Anda. Sesuaikan jika perlu.
    # Ini adalah NAMA KOLOM ASLI dari df_logs (setelah parsing)
    categorical_features_to_use = [
        'srcip', 'srccountry', 'action', 'proto', 'service', 'level', 'app', 
        'appcat', 'crlevel', 'policyid', 'policytype', 'subtype', 'dstcountry', 
        'srcintf', 'dstintf', 'vd', 'type', 'trandisp', 'devname'
    ]
    numerical_features_to_use = [
        'duration', 'sentbyte', 'rcvdbyte', 'sentpkt', 'rcvdpkt', 
        'crscore', 'srcport', 'dstport', 'sessionid' 
    ]

    all_categorical_to_consider = sorted(list(set(categorical_features_to_use)))
    all_numerical_to_consider = sorted(list(set(numerical_features_to_use)))
    
    available_categorical_original = [f for f in all_categorical_to_consider if f in df_logs.columns]
    available_numerical_original = [f for f in all_numerical_to_consider if f in df_logs.columns]

    print(f"\nFitur Kategorikal Asli yang akan diproses: {available_categorical_original}")
    print(f"Jumlah: {len(available_categorical_original)}")
    print(f"Fitur Numerik Asli yang akan diproses: {available_numerical_original}")
    print(f"Jumlah: {len(available_numerical_original)}")

    # === 2. PROSES FITUR KATEGORIKAL ===
    df_cat_processed = pd.DataFrame(index=df_logs.index)
    label_encoders = {}
    if available_categorical_original:
        print("\n--- Memproses Fitur Kategorikal ---")
        for col in available_categorical_original:
            s = df_logs[col].astype(str).fillna('Unknown').replace('', 'Unknown')
            le = LabelEncoder()
            df_cat_processed[col] = le.fit_transform(s)
            label_encoders[col] = le
            # print(f"  Encoded: {col}") # Bisa di-uncomment untuk detail
        print(f"✅ Fitur kategorikal ({len(available_categorical_original)} kolom) telah di-encode.")
    else:
        print("ℹ️ Tidak ada fitur kategorikal yang tersedia/dipilih.")

    # === 3. PROSES FITUR NUMERIK ===
    df_num_processed = pd.DataFrame(index=df_logs.index)
    if available_numerical_original:
        print("\n--- Memproses Fitur Numerik ---")
        for col in available_numerical_original:
            s_num = pd.to_numeric(df_logs[col], errors='coerce')
            median_val = s_num.median()
            s_num_filled = s_num.fillna(median_val if pd.notna(median_val) else 0)
            df_num_processed[col] = s_num_filled
            # print(f"  Processed: {col} (NaNs filled with {median_val if pd.notna(median_val) else 0:.2f})") # Bisa di-uncomment
        print(f"✅ Fitur numerik ({len(available_numerical_original)} kolom) telah diproses.")
    else:
        print("ℹ️ Tidak ada fitur numerik yang tersedia/dipilih.")

    # === 4. GABUNGKAN FITUR ===
    df_list_to_concat = []
    # Gabungkan dengan urutan: semua kolom kategorikal (sudah di-encode), lalu semua kolom numerik.
    # Urutan diambil dari available_categorical_original dan available_numerical_original yang sudah di-sort.
    if not df_cat_processed.empty: df_list_to_concat.append(df_cat_processed[available_categorical_original])
    if not df_num_processed.empty: df_list_to_concat.append(df_num_processed[available_numerical_original])
    
    df_combined_processed = pd.DataFrame() # Inisialisasi
    if df_list_to_concat: df_combined_processed = pd.concat(df_list_to_concat, axis=1)
    
    df_combined_processed_cols = [] 
    if df_combined_processed.empty:
        print("\n❌ DataFrame gabungan kosong. Tidak ada fitur yang bisa diproses lebih lanjut.")
        df_processed = pd.DataFrame()
    else:
        df_combined_processed_cols = df_combined_processed.columns.tolist()
        print(f"\n✅ Fitur telah digabungkan. Total fitur: {df_combined_processed.shape[1]}.")
        print(f"Urutan kolom untuk scaler: {df_combined_processed_cols}")
        # print("\n📋 Pratinjau DataFrame Gabungan (5 baris pertama sebelum scaling):")
        # print(df_combined_processed.head())
        
        # === 5. SCALING ===
        print("\n⏳ Menerapkan MinMaxScaler...")
        scaler = MinMaxScaler()
        df_scaled_values = scaler.fit_transform(df_combined_processed)
        df_processed = pd.DataFrame(df_scaled_values, columns=df_combined_processed_cols, index=df_combined_processed.index)
        print("✅ Semua fitur telah diskalakan.")
        print("\n📋 Pratinjau DataFrame Siap Latih (df_processed):")
        print(df_processed.head())

# Inisialisasi fallback jika df_processed tidak terbuat karena df_logs kosong
if 'df_processed' not in locals(): df_processed = pd.DataFrame()
if 'label_encoders' not in locals(): label_encoders = {}
if 'scaler' not in locals(): scaler = None # Akan error jika training dijalankan dengan scaler=None
if 'available_categorical_original' not in locals(): available_categorical_original = []
if 'available_numerical_original' not in locals(): available_numerical_original = []
if 'df_combined_processed_cols' not in locals(): df_combined_processed_cols = []


# Sel 5: Melatih Model Autoencoder
# =============================================================================
# ### 5. Melatih Model Autoencoder 🤖
# =============================================================================
autoencoder = None 
history_ae = None 
if df_processed.empty:
    print("\n❌ DataFrame 'df_processed' kosong. Tidak dapat melatih Autoencoder.")
else:
    print("\n--- Melatih Model Autoencoder ---")
    input_dim_ae = df_processed.shape[1]
    if input_dim_ae == 0:
        print("❌ Dimensi input Autoencoder adalah 0. Tidak bisa melatih.")
    else:
        # Arsitektur Autoencoder (contoh, sesuaikan dengan jumlah fitur baru Anda)
        # Anda bisa membuat ini lebih dinamis atau menetapkannya berdasarkan input_dim_ae
        l1_nodes = max(10, int(input_dim_ae * 0.75)) if input_dim_ae > 0 else 10
        l2_nodes_bottleneck = max(8, int(input_dim_ae * 0.50)) if input_dim_ae > 0 else 8
        if l2_nodes_bottleneck >= l1_nodes and l1_nodes > 1 : l2_nodes_bottleneck = max(4, int(l1_nodes/2))


        print(f"Arsitektur AE: Input({input_dim_ae}) -> Dense({l1_nodes}) -> Dropout(0.2) -> Dense({l2_nodes_bottleneck}) [Bottleneck] -> Dense({l1_nodes}) -> Dropout(0.2) -> Output({input_dim_ae})")
        
        input_layer_ae = Input(shape=(input_dim_ae,))
        encoder = Dense(l1_nodes, activation="relu")(input_layer_ae)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(l2_nodes_bottleneck, activation="relu")(encoder) # Bottleneck
        
        decoder = Dense(l1_nodes, activation="relu")(encoder)
        decoder = Dropout(0.2)(decoder)
        decoder_output = Dense(input_dim_ae, activation='sigmoid')(decoder)
        
        autoencoder = Model(inputs=input_layer_ae, outputs=decoder_output)
        autoencoder.compile(optimizer='adam', loss='mse') # mse adalah singkatan dari mean_squared_error
        autoencoder.summary()
        
        output_dir_models = 'trained_models_artifacts'
        os.makedirs(output_dir_models, exist_ok=True) # Pastikan folder ada
        autoencoder_save_path = os.path.join(output_dir_models, 'autoencoder_model.keras')
        
        early_stopping_ae = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        model_checkpoint_ae = ModelCheckpoint(filepath=autoencoder_save_path, save_best_only=True, monitor='val_loss', verbose=1)
        
        print("\n⏳ Memulai pelatihan Autoencoder...")
        history_ae = autoencoder.fit(df_processed, df_processed, 
                                  epochs=50, # Kurangi epoch untuk iterasi lebih cepat, bisa dinaikkan lagi
                                  batch_size=32, # Batch size lebih kecil mungkin membantu generalisasi
                                  shuffle=True, 
                                  validation_split=0.2, 
                                  callbacks=[early_stopping_ae, model_checkpoint_ae], 
                                  verbose=1)
        print("\n✅ Pelatihan Autoencoder Selesai.")
        
        print(f"Memuat model Autoencoder terbaik dari: {autoencoder_save_path}")
        try:
            autoencoder = load_model(autoencoder_save_path) 
            print("✅ Model terbaik Autoencoder berhasil dimuat ulang.")
        except Exception as e:
            print(f"⚠️ Gagal memuat model terbaik dari checkpoint: {e}. Model terakhir dari .fit() akan digunakan.")
        
        plt.figure(figsize=(10,6))
        plt.plot(history_ae.history['loss'], label='Training Loss (MSE)')
        plt.plot(history_ae.history['val_loss'], label='Validation Loss (MSE)')
        plt.title('Kurva Loss Autoencoder'); plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True); plt.show()

# Sel 6: Penyimpanan MSE Training AE
# =============================================================================
# ### 6. Penyimpanan MSE Training Autoencoder
# =============================================================================
mse_train_ae = np.array([]) # Inisialisasi sebagai array kosong
if autoencoder is not None and not df_processed.empty:
    print("\n--- Menghitung dan Menyimpan Reconstruction Error (MSE) Training Autoencoder ---")
    reconstructions_ae = autoencoder.predict(df_processed)
    # Pastikan df_processed adalah numpy array atau konversi jika masih DataFrame
    df_processed_np = df_processed.to_numpy() if isinstance(df_processed, pd.DataFrame) else df_processed
    mse_train_ae = np.mean(np.power(df_processed_np - reconstructions_ae, 2), axis=1)
    
    output_dir_artifacts = 'trained_models_artifacts' # Pastikan ini konsisten
    os.makedirs(output_dir_artifacts, exist_ok=True)
    mse_training_save_path = os.path.join(output_dir_artifacts, 'training_mse_ae.npy')
    try:
        np.save(mse_training_save_path, mse_train_ae)
        print(f"💾 Training MSE Autoencoder berhasil disimpan ke: {mse_training_save_path}")
    except Exception as e:
        print(f"❌ Gagal menyimpan Training MSE Autoencoder: {e}")

    # Visualisasi distribusi MSE training (opsional di sini, karena sudah ada di dashboard)
    # threshold_ae_train = np.percentile(mse_train_ae, 95) 
    # plt.figure(figsize=(10,6)); sns.histplot(mse_train_ae, bins=50, kde=True); plt.axvline(threshold_ae_train, color='r', linestyle='--'); plt.title('Distribusi MSE Training AE'); plt.show()
    # print(f"Contoh threshold (95-percentile) dari MSE training: {threshold_ae_train:.6f}")
else:
    print("\n⚠️ Autoencoder tidak dilatih atau df_processed kosong, MSE training tidak dihitung/disimpan.")


# Sel 7: Melatih Model One-Class SVM
# =============================================================================
# ### 7. Melatih Model One-Class SVM (OCSVM) 🛡️
# =============================================================================
ocsvm = None # Inisialisasi
if df_processed.empty:
    print("\n❌ DataFrame 'df_processed' kosong. Tidak dapat melatih One-Class SVM.")
else:
    print("\n--- Melatih Model One-Class SVM ---")
    # Anda bisa eksperimen dengan nilai nu. Nilai yang lebih kecil membuat model lebih sensitif (lebih banyak anomali).
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05) 
    print(f"\n⏳ Memulai pelatihan OCSVM dengan parameter: kernel='rbf', gamma='auto', nu=0.05...")
    ocsvm.fit(df_processed) # Dilatih hanya dengan data normal
    print("✅ Pelatihan One-Class SVM Selesai.")
    
    output_dir_models = 'trained_models_artifacts' # Konsistenkan nama folder
    os.makedirs(output_dir_models, exist_ok=True)
    ocsvm_save_path = os.path.join(output_dir_models, 'ocsvm_model.pkl')
    joblib.dump(ocsvm, ocsvm_save_path)
    print(f"💾 Model OCSVM disimpan ke: {ocsvm_save_path}")

# Sel 8: Menyimpan Semua Artefak Pra-pemrosesan (Final)
# =============================================================================
# ### 8. Menyimpan Semua Artefak Pra-pemrosesan (Final) 📦
# =============================================================================
output_dir_artifacts = 'trained_models_artifacts' 
os.makedirs(output_dir_artifacts, exist_ok=True)

print(f"\n📁 Menyimpan semua artefak pra-pemrosesan yang relevan ke: {output_dir_artifacts}")

if label_encoders: 
    le_path = os.path.join(output_dir_artifacts, 'label_encoders.pkl')
    joblib.dump(label_encoders, le_path)
    print(f"💾 Label Encoders disimpan ke: {le_path} (Total: {len(label_encoders)} encoders)")
else:
    print("ℹ️ Tidak ada Label Encoders untuk disimpan (kemungkinan tidak ada fitur kategorikal).")

if scaler is not None: 
    sc_path = os.path.join(output_dir_artifacts, 'scaler.pkl')
    joblib.dump(scaler, sc_path)
    print(f"💾 Scaler disimpan ke: {sc_path}")
else:
    print("ℹ️ Tidak ada Scaler untuk disimpan (kemungkinan pra-pemrosesan tidak sampai tahap scaling).")

if df_combined_processed_cols: 
    model_cols_path = os.path.join(output_dir_artifacts, 'model_columns.pkl')
    joblib.dump(df_combined_processed_cols, model_cols_path)
    print(f"💾 Daftar kolom model (untuk scaler) disimpan ke: {model_cols_path} ({len(df_combined_processed_cols)} kolom)")
else:
    print("ℹ️ Tidak ada daftar kolom model untuk disimpan.")

# Simpan feature_types jika ada (nama kolom asli kategorikal dan numerik)
feature_types_to_save = {
    'categorical_original_names': available_categorical_original if 'available_categorical_original' in locals() else [],
    'numerical_original_names': available_numerical_original if 'available_numerical_original' in locals() else []
}
if feature_types_to_save['categorical_original_names'] or feature_types_to_save['numerical_original_names']:
    ft_path = os.path.join(output_dir_artifacts, 'feature_types.pkl')
    joblib.dump(feature_types_to_save, ft_path)
    print(f"💾 Tipe fitur (nama kolom asli) disimpan ke: {ft_path}")
else:
    print("ℹ️ Tidak ada tipe fitur untuk disimpan.")

print("\n✨ Penyimpanan artefak pra-pemrosesan selesai.")

# Sel 9: Selesai!
# =============================================================================
# ### 9. Selesai! 🎉
# =============================================================================
print("\nProses training dengan rekayasa fitur baru dan penyimpanan model/artefak selesai.")
print("Model dan artefak siap digunakan oleh 'pages/1_Dashboard.py'.")
print("Pastikan untuk memverifikasi semua artefak di folder 'trained_models_artifacts'.")
