# streamlit_app.py
import streamlit as st
import os
# import time # Tidak digunakan di kode Anda sebelumnya, bisa dihilangkan jika tidak perlu

# --- Konfigurasi Dasar ---
# Kredensial Login akan diambil dari st.secrets saat di Streamlit Cloud
# Untuk lokal, Anda bisa membuat file .streamlit/secrets.toml
# Contoh .streamlit/secrets.toml:
# VALID_USERNAME = "analis"
# VALID_PASSWORD = "password_rahasia_anda"

# Path ke artefak model tidak secara langsung digunakan di file ini,
# tapi didefinisikan di 1_Dashboard.py dan train_script.ipynb.
# BASE_DIR = os.path.abspath(os.path.dirname(__file__)) # Ini sudah benar
# MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts') # Tidak perlu di sini

# --- Fungsi Autentikasi ---
def check_login(username, password):
    """Memeriksa kredensial login menggunakan st.secrets."""
    try:
        # Ambil kredensial dari Streamlit secrets (untuk deployment)
        valid_user = st.secrets["VALID_USERNAME"]
        valid_pass = st.secrets["VALID_PASSWORD"]
    except KeyError:
        # Fallback jika st.secrets tidak terkonfigurasi (untuk development lokal)
        # PENTING: JANGAN GUNAKAN KREDENSIAL DEFAULT INI DI PRODUKSI!
        # Buat file .streamlit/secrets.toml untuk lokal.
        st.warning("Kredensial admin (VALID_USERNAME/VALID_PASSWORD) belum dikonfigurasi di st.secrets. Menggunakan kredensial default untuk development.")
        valid_user = "analis"  # Ganti dengan username default Anda untuk lokal
        valid_pass = "password123" # Ganti dengan password default Anda untuk lokal
        
    return username == valid_user and password == valid_pass

def show_login_form():
    """Menampilkan form login."""
    # Anda bisa mengganti placeholder ini dengan gambar logo atau header aplikasi Anda
    # st.image("https://placehold.co/600x150/2D3748/E2E8F0?text=Sistem+Deteksi+Anomali+Akses", use_column_width=True) 
    st.markdown("<h1 style='text-align: center;'>Sistem Deteksi Anomali Akses Jaringan</h1>", unsafe_allow_html=True)
    st.markdown("---")
    # st.title("Login Security Analyst") # Sudah ada di markdown di atas

    with st.form("login_form"):
        st.subheader("Silakan Login")
        username = st.text_input("Username", key="login_username_input", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", key="login_password_input", placeholder="Masukkan password")
        submitted = st.form_submit_button("Login 🔑")

        if submitted:
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                if "login_error" in st.session_state: # Hapus error sebelumnya jika ada
                    del st.session_state["login_error"]
                st.success(f"Login berhasil sebagai {username}!")
                st.rerun() # Jalankan ulang skrip untuk merefleksikan status login
            else:
                st.session_state["login_error"] = "Username atau password salah."
                # Pesan error akan ditampilkan di bawah form

    # Tampilkan pesan error di luar form agar tetap terlihat setelah submit gagal
    if "login_error" in st.session_state and st.session_state["login_error"]:
        st.error(st.session_state["login_error"])
        del st.session_state["login_error"] # Hapus error setelah ditampilkan


# --- Logika Aplikasi Utama ---
def main():
    st.set_page_config(
        page_title="Deteksi Anomali Akses", 
        page_icon="🛡️", # Tambahkan page_icon jika diinginkan
        layout="centered", 
        initial_sidebar_state="collapsed" 
    )

    # Inisialisasi session state jika belum ada
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    # Flag ini mungkin tidak lagi diperlukan jika pesan sambutan diubah
    # if "dashboard_loaded_once" not in st.session_state: 
    #     st.session_state["dashboard_loaded_once"] = False


    if not st.session_state["logged_in"]:
        show_login_form()
    else:
        # Jika sudah login, tampilkan pesan dan biarkan sidebar untuk navigasi
        # Pesan sambutan bisa lebih sederhana atau langsung mengarahkan
        st.sidebar.success(f"Login sebagai: {st.session_state['username']}")
        st.sidebar.markdown("---") 
        
        st.title(f"Selamat Datang di Sistem Deteksi Anomali, {st.session_state['username']}!")
        st.info(
            "Silakan pilih **'1_Dashboard'** dari menu navigasi di sebelah kiri (sidebar) "
            "untuk memulai proses deteksi anomali pada log Fortigate.", 
            icon="👈"
        )
        st.markdown(
            "Aplikasi ini membantu Anda menganalisis log jaringan untuk "
            "mengidentifikasi aktivitas yang berpotensi mencurigakan menggunakan model "
            "Autoencoder dan One-Class SVM."
        )
        st.markdown("---")
        
        # Tombol logout bisa juga diletakkan di sini jika diinginkan selain di sidebar dashboard
        if st.sidebar.button("Logout", key="main_logout_button_sidebar_app", use_container_width=True):
            # Reset semua session state yang relevan saat logout
            keys_to_delete_on_logout = [
                "logged_in", "username", "login_error", 
                "detection_output", "models_artifacts_loaded" # Tambahkan state dari dashboard
            ]
            for key in keys_to_delete_on_logout:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()