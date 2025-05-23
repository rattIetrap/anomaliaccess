# streamlit_app.py
import streamlit as st
import os
import time # Untuk simulasi loading jika diperlukan

# --- Konfigurasi Dasar ---
# Kredensial Login akan diambil dari st.secrets saat di Streamlit Cloud
# Untuk lokal, Anda bisa membuat file .streamlit/secrets.toml
# Contoh .streamlit/secrets.toml:
# VALID_USERNAME = "analis"
# VALID_PASSWORD = "password_rahasia_anda"

# Path ke artefak model (relatif terhadap direktori streamlit_app.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'trained_models_artifacts')

# --- Fungsi Autentikasi ---
def check_login(username, password):
    """Memeriksa kredensial login menggunakan st.secrets."""
    try:
        valid_user = st.secrets["VALID_USERNAME"]
        valid_pass = st.secrets["VALID_PASSWORD"]
    except KeyError:
        st.error("Kredensial admin (VALID_USERNAME/VALID_PASSWORD) belum dikonfigurasi di st.secrets aplikasi Streamlit.")
        # Fallback ke nilai default jika tidak ada di secrets (TIDAK DIREKOMENDASIKAN UNTUK PRODUKSI)
        # Ini hanya untuk memudahkan pengujian lokal jika secrets.toml belum dibuat
        # Hapus fallback ini di produksi.
        print("PERINGATAN: Menggunakan kredensial fallback karena st.secrets tidak terkonfigurasi.")
        valid_user = "analis" 
        valid_pass = "password123"
        
    return username == valid_user and password == valid_pass

def show_login_form():
    """Menampilkan form login."""
    # Anda bisa mengganti placeholder ini dengan gambar yang lebih relevan
    st.image("https://placehold.co/600x150/2D3748/E2E8F0?text=Sistem+Deteksi+Anomali+Akses", use_column_width=True) 
    st.title("Login Security Analyst")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username_input", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", key="login_password_input", placeholder="Masukkan password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["login_error"] = None # Hapus error sebelumnya
                st.rerun() # Jalankan ulang skrip untuk merefleksikan status login
            else:
                st.session_state["login_error"] = "Username atau password salah."
                # Error akan ditampilkan di bawah form

    # Tampilkan pesan error di luar form agar tetap terlihat setelah submit gagal
    if "login_error" in st.session_state and st.session_state["login_error"]:
        st.error(st.session_state["login_error"])


# --- Logika Aplikasi Utama ---
def main():
    st.set_page_config(
        page_title="Deteksi Anomali Akses", 
        layout="centered", # 'centered' atau 'wide'
        initial_sidebar_state="collapsed" # 'auto', 'expanded', 'collapsed'
    )

    # Inisialisasi session state jika belum ada
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "dashboard_loaded_once" not in st.session_state: # Flag untuk pesan sambutan
        st.session_state["dashboard_loaded_once"] = False


    if not st.session_state["logged_in"]:
        show_login_form()
    else:
        # Jika sudah login, tampilkan pesan dan biarkan sidebar untuk navigasi
        st.sidebar.success(f"Login sebagai: {st.session_state['username']}")
        st.sidebar.markdown("---") # Garis pemisah
        
        st.title(f"Selamat Datang, {st.session_state['username']}!")
        st.info("Silakan pilih **'1_Dashboard'** dari menu di sebelah kiri untuk memulai proses deteksi anomali.", icon="👈")
        st.markdown("Aplikasi ini membantu Anda menganalisis log Fortigate untuk mendeteksi aktivitas yang mencurigakan.")
        
        if st.sidebar.button("Logout", key="main_logout_button_sidebar", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["dashboard_loaded_once"] = False # Reset flag
            if "login_error" in st.session_state: 
                del st.session_state["login_error"]
            # Hapus state spesifik dashboard saat logout
            if "results_df" in st.session_state:
                del st.session_state["results_df"]
            if "last_file_name" in st.session_state:
                del st.session_state["last_file_name"]
            if "last_unique_id" in st.session_state:
                del st.session_state["last_unique_id"]
            st.rerun()

if __name__ == "__main__":
    main()