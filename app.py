import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import time
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import textwrap

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS PREMIUM
# ==========================================
st.set_page_config(
    page_title="EduNews Search - Mesin Pencari Vertikal Berita Pendidikan",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download Resource NLTK (Silent Mode)
try: nltk.data.find('corpora/stopwords')
except LookupError: nltk.download('stopwords')

# --- CUSTOM CSS (TAMPILAN PROFESIONAL & MODERN) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #0f172a !important;
    }

    /* LANDING PAGE STYLES */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 80px 40px;
        border-radius: 24px;
        text-align: center;
        margin: 40px 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
        font-family: 'Poppins', sans-serif;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: rgba(255,255,255,0.9);
        margin-bottom: 40px;
        font-weight: 500;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* TEAM SECTION */
    .team-section {
        margin: 30px 0 20px 0;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    }

    .team-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 2px;
        font-family: 'Poppins', sans-serif;
    }

    /* Style untuk gambar tim yang di-embed base64 */
    .team-section img {
        display: block;
        margin: 0 auto 20px auto;
    }

    .team-name {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 15px;
        margin-bottom: 8px;
        font-family: 'Poppins', sans-serif;
        text-align: center;
    }

    .team-role {
        font-size: 14px;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
    }
    
    .feature-card {
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 20px;
        display: block;
    }
    
    .feature-title {
        font-size: 22px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        font-family: 'Poppins', sans-serif;
    }
    
    .feature-desc {
        font-size: 15px;
        color: #64748b;
        line-height: 1.6;
    }

    /* SIDEBAR PREMIUM STYLING */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #0f1419 100%);
        border-right: 2px solid #2d3748;
        box-shadow: 4px 0 20px rgba(0,0,0,0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #f7fafc !important;
    }
    
    /* Logo Section */
    [data-testid="stSidebar"] .sidebar-logo {
        text-align: center;
        padding: 25px 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    
    /* Menu Items */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 8px;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 14px 20px !important;
        margin-bottom: 6px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid transparent !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
        transform: translateX(5px);
        cursor: pointer;
    }
    
    /* Active Menu */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar Stats */
    [data-testid="stSidebar"] .sidebar-stat {
        background: rgba(255,255,255,0.08);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #667eea;
        font-weight: 600;
    }
    
    /* Sidebar Footer */
    [data-testid="stSidebar"] .sidebar-footer {
        position: absolute;
        bottom: 20px;
        left: 0;
        right: 0;
        padding: 15px 20px;
        text-align: center;
        border-top: 1px solid rgba(255,255,255,0.1);
        font-size: 12px;
        color: #a0aec0 !important;
    }

    /* METRIC CARDS */
    .metric-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        text-align: center;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }
    
    .metric-val {
        font-size: 42px;
        font-weight: 800;
        color: #0f172a;
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-lbl {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: #f8fafc;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        color: #64748b;
        font-size: 16px;
        font-weight: 600;
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* SEARCH RESULTS */
    .result-container {
        background: white;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .result-container:hover {
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.2);
        transform: translateY(-3px);
    }
    
    .result-score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        margin-bottom: 15px;
        border: 1px solid #93c5fd;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .result-link a {
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #1e40af;
        text-decoration: none;
        line-height: 1.4;
        display: block;
        margin-bottom: 10px;
    }
    
    .result-link a:hover {
        text-decoration: underline;
        color: #1e3a8a;
    }
    
    .result-meta {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 16px;
        display: flex;
        gap: 15px;
        align-items: center;
        font-weight: 500;
    }
    
    .result-snippet {
        font-size: 15px;
        color: #475569;
        line-height: 1.8;
        padding-top: 12px;
        border-top: 1px solid #f1f5f9;
    }

    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        height: 55px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        color: white;
    }

    /* HIGHLIGHT */
    .highlight {
        background-color: #fef08a;
        color: #854d0e;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
    }
    
    /* INPUT STYLING */
    div[data-baseweb="input"] {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        padding: 6px;
        background-color: white;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* START BUTTON */
    .start-button {
        display: inline-block;
        background: white;
        color: #667eea;
        padding: 16px 40px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 18px;
        text-decoration: none;
        transition: all 0.3s;
        border: 2px solid white;
        cursor: pointer;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .start-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    
    /* STATS BAR */
    .stats-bar {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 30px 0;
        text-align: center;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ==================
# 2. FUNGSI BACKEND 
# ==================

@st.cache_resource
def get_resources():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    custom_stop = {'baca', 'juga', 'halaman', 'kompas', 'detik', 'com', 'wib', 'jakarta', 
                   'terkait', 'untuk', 'pada', 'adalah', 'yang', 'dan', 'di', 'ke', 'dari', 
                   'ini', 'itu', 'dengan', 'dalam', 'tribun', 'liputan6', 'penulis', 'editor'}
    stop_words.update(custom_stop)
    return stemmer, stop_words

def preprocess_text(text, stemmer, stop_words):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    clean_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 3]
    return " ".join(clean_tokens)

@st.cache_data
def load_dataset():
    file_nb = 'Lampiran_Data_Bersih.csv'
    file_meta = 'Lampiran_Data_Mentah.csv'
    
    if os.path.exists(file_nb) and os.path.exists(file_meta):
        try:
            df_nb = pd.read_csv(file_nb)
            if len(df_nb) > 50: 
                df_meta = pd.read_csv(file_meta)
                if 'Doc_ID' in df_meta.columns and 'Doc_ID' in df_nb.columns:
                    df = pd.merge(df_meta, df_nb[['Doc_ID', 'Clean_Content']], on='Doc_ID', how='left')
                    df['Clean_Content'] = df['Clean_Content'].fillna('')
                    return df, "Cache Notebook (Fast Load)"
        except Exception: pass

    try:
        raw_file = 'korpus_pendidikan_gabungan.csv'
        if not os.path.exists(raw_file):
            return pd.DataFrame(), "File CSV Tidak Ditemukan"
            
        cleaned_rows = []
        with open(raw_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try: next(reader)
            except: pass
            for row in reader:
                while row and row[-1] == '': row.pop()
                if not row: continue
                url_idx = -1
                for i, col in enumerate(row):
                    if 'http' in col: url_idx = i; break
                if url_idx != -1:
                    cleaned_rows.append([row[0], ",".join(row[1:url_idx-1]).strip('"'), row[url_idx-1].strip(), row[url_idx].strip(), ",".join(row[url_idx+1:-1]).strip('"'), row[-1].strip('"')])
                    
        df_raw = pd.DataFrame(cleaned_rows, columns=['Doc_ID', 'Title', 'Source', 'URL', 'Date', 'Content'])
        
        stemmer, stop_words = get_resources()
        df_raw['Clean_Content'] = df_raw['Content'].apply(lambda x: preprocess_text(str(x), stemmer, stop_words))
        return df_raw, "Processed Raw (Full Data)"
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

@st.cache_resource
def build_engine(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return vectorizer, tfidf_matrix, bm25

def highlight_text(text, query):
    if not query or not isinstance(text, str): return text
    terms = sorted(query.lower().split(), key=len, reverse=True)
    for term in terms:
        if len(term) > 2:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<span class='highlight'>{m.group(0)}</span>", text)
    return text

def get_snippet(text, query):
    if not isinstance(text, str): return ""
    start = text.lower().find(query.lower().split()[0]) if query else -1
    if start == -1: return text[:160] + "..."
    start = max(0, start - 60); end = min(len(text), start + 160)
    return ("..." if start>0 else "") + text[start:end] + "..."

# ==========================================
# 3. SESSION STATE MANAGEMENT
# ==========================================
if 'app_started' not in st.session_state:
    st.session_state.app_started = False

# ==========================================
# 4. LANDING PAGE
# ==========================================
if not st.session_state.app_started:
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🎓 EduNews Search</div>
        <div class="hero-subtitle">
            Mesin Pencari Vertikal untuk Berita Pendidikan Berbahasa Indonesia<br>
            Menggunakan Algoritma TF-IDF dan BM25<br>
            Pencarian cepat, akurat, dan relevan untuk informasi pendidikan
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <div class="feature-title">Pencarian Cerdas</div>
            <div class="feature-desc">Algoritma TF-IDF & BM25 untuk hasil pencarian yang akurat</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Analisis Visual</div>
            <div class="feature-desc">Dashboard interaktif dengan visualisasi data yang mendalam</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚙️</span>
            <div class="feature-title">Evaluasi Kinerja</div>
            <div class="feature-desc">Matrix evaluasi untuk mengukur akurasi sistem retrieval</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📚</span>
            <div class="feature-title">Dataset Lengkap</div>
            <div class="feature-desc">Korpus berita pendidikan Indonesia yang komprehensif</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Team Section
    def load_base64(path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()

    # convert gambar
    img_dea = load_base64("foto/dea.jpg")
    img_tasya = load_base64("foto/tasya.jpg")
    img_dinda = load_base64("foto/dinda.jpg")

    # wrapper foto
    image_wrapper = """
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <img src="data:image/jpeg;base64,{img}" 
            style="width: 250px; height: 250px; border-radius: 50%; object-fit: cover;
            border: 6px solid #667eea; box-shadow: 0 12px 35px rgba(102, 126, 234, 0.35);" />
    </div>
    """
    
    st.markdown("""
    <div class="team-section">
        <div class="team-title">👥 Tim Pengembang</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # TEAM 1
    with col1:
        st.markdown(image_wrapper.format(img=img_dea), unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center;">
                <div class="team-name">Dea Zasqia Pasaribu Malau</div>
                <div class="team-role">2308107010004</div>
            </div>
        """, unsafe_allow_html=True)

    # TEAM 2
    with col2:
        st.markdown(image_wrapper.format(img=img_tasya), unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center;">
                <div class="team-name">Tasya Zahrani</div>
                <div class="team-role">2308107010006</div>
            </div>
        """, unsafe_allow_html=True)

    # TEAM 3
    with col3:
        st.markdown(image_wrapper.format(img=img_dinda), unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center;">
                <div class="team-name">Adinda Muarriva</div>
                <div class="team-role">2308107010001</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Start Button
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if st.button("🚀 MULAI PENCARIAN", use_container_width=True, type="primary"):
            st.session_state.app_started = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats Preview
    with st.spinner("📥 Memuat preview data..."):
        df_preview, _ = load_dataset()
        if not df_preview.empty:

            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-top: 10px;">
                    <div style="
                        padding: 18px 28px;
                        background: linear-gradient(to bottom right, #ffffff, #eef3f9);
                        border-radius: 15px;
                        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
                        text-align: center;
                        font-size: 17px;
                        max-width: 520px;
                    ">
                        <b>📈 Preview Dataset:</b> {len(df_preview)} Dokumen Siap Dianalisis |
                        <b>🏢 Sumber:</b> {df_preview['Source'].nunique() if 'Source' in df_preview.columns else 'N/A'} Media |
                        <b>🗓️</b> Data Terkini
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 14px; padding: 20px;'>
        <b>© 2025 Kelompok 10 - Teknologi Pencarian Informasi</b><br>
        Universitas Syiah Kuala • Fakultas MIPA • Informatika<br>
        <small>Tugas Akhir: Mesin Pencari Vertikal Berita Pendidikan</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# ==========================================
# 5. MAIN APPLICATION (AFTER START)
# ==========================================

# Load Data
with st.spinner("🚀 Memuat Data & AI Engine..."):
    df, status_msg = load_dataset()
    if df.empty:
        st.error(f"Gagal memuat data. Pesan: {status_msg}")
        st.stop()
    df['Clean_Content'] = df['Clean_Content'].fillna('').astype(str)
    vectorizer, tfidf_matrix, bm25 = build_engine(df['Clean_Content'].tolist())
    stemmer, stop_words = get_resources()

# =========================
#       SIDEBAR UTAMA
# =========================

with st.sidebar:
    # CSS untuk hilangkan sidebar nav
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    
    /* Style untuk menu custom */
    .menu-item {
        background: rgba(255,255,255,0.05);
        padding: 12px 20px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #94a3b8;
        font-size: 15px;
    }
    .menu-item:hover {
        background: rgba(102, 126, 234, 0.2);
        color: white;
    }
    .menu-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:-10px;'>
        <span style="font-size:60px;">🎓</span>
        <h1 style="color:white; font-size:26px; margin-bottom:-5px;">
            EduNews Search
        </h1>
        <p style="color:#94a3b8; font-size:14px; margin-top:5px;">
            Mesin Pencari Vertikal Berita Pendidikan
        </p>
        <hr style="border-color:#334155; margin-top:20px; margin-bottom:20px;">
    </div>
    """, unsafe_allow_html=True)

    # Initialize menu state
    if 'menu' not in st.session_state:
        st.session_state.menu = "🏠 Dashboard"
    
    # Menu buttons
    if st.button("🏠 Dashboard", key="dash", use_container_width=True):
        st.session_state.menu = "🏠 Dashboard"
    
    if st.button("🔍 Mesin Pencari", key="search", use_container_width=True):
        st.session_state.menu = "🔍 Mesin Pencari"
    
    if st.button("⚙️ Evaluasi Kinerja", key="eval", use_container_width=True):
        st.session_state.menu = "⚙️ Evaluasi Kinerja"
    
    if st.button("📂 Dataset Korpus", key="data", use_container_width=True):
        st.session_state.menu = "📂 Dataset Korpus"
    
    if st.button("ℹ️ Tentang", key="about", use_container_width=True):
        st.session_state.menu = "ℹ️ Tentang"
    
    menu = st.session_state.menu

    st.markdown("<hr style='border-color:#334155; margin-top:25px; margin-bottom:25px;'>", unsafe_allow_html=True)

    # Statistics
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.1); padding:15px; border-radius:10px; margin-bottom:15px;'>
        <div style='font-size:11px; color:#a0aec0; margin-bottom:6px; letter-spacing:0.5px;'>TOTAL DOKUMEN</div>
        <div style='font-size:24px; font-weight:700; color:#667eea;'>📄 {len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Source' in df.columns:
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.1); padding:15px; border-radius:10px; margin-bottom:20px;'>
            <div style='font-size:11px; color:#a0aec0; margin-bottom:6px; letter-spacing:0.5px;'>SUMBER MEDIA</div>
            <div style='font-size:24px; font-weight:700; color:#667eea;'>🏢 {df['Source'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Back to Home Button
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.app_started = False
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style='margin-top: 40px; text-align:center; 
                font-size:11px; color:#64748b; padding-top:20px; 
                border-top:1px solid rgba(255,255,255,0.1);'>
        © 2025 Kelompok 10<br>
        <span style='color:#667eea; font-weight:600;'>Teknologi Pencarian Informasi</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. HALAMAN KONTEN (SAMA SEPERTI SEBELUMNYA)
# ==========================================

# --- PAGE 1: DASHBOARD ---
if menu == "🏠 Dashboard":
    st.markdown("<h2 style='margin-bottom:10px;'>📊 Dashboard Analisis Korpus</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:30px;'>Statistik visual mendalam mengenai distribusi kata dan topik dalam korpus berita pendidikan.</p>", unsafe_allow_html=True)
    
    all_text = " ".join(df['Clean_Content'])
    doc_lens = df['Clean_Content'].apply(lambda x: len(x.split()))
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-lbl">Total Artikel</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="metric-val">{int(doc_lens.mean())}</div><div class="metric-lbl">Rata-rata Kata</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(set(all_text.split())):,}</div><div class="metric-lbl">Kosakata Unik</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["☁️ WordCloud Besar", "📊 Tren Kata & Frasa", "📉 Distribusi Data"])
    
    sns.set_style("whitegrid")
    PALETTE = "viridis"
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
 
    with tab1:
        st.subheader("Kata Kunci Paling Dominan")
        wc = WordCloud(width=1600, height=700, background_color='white', colormap=PALETTE, max_words=150, contour_width=0, stopwords=set(), collocations=False).generate(all_text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig)
        
    with tab2:
        c_uni, c_bi = st.columns(2)
        with c_uni:
            st.markdown("**Top 15 Kata (Unigram)**")
            counter = Counter(all_text.split())
            df_uni = pd.DataFrame(counter.most_common(15), columns=['Kata', 'Frekuensi'])
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=df_uni, x='Frekuensi', y='Kata', hue='Kata', palette=PALETTE, ax=ax1, legend=False)
            ax1.set_xlabel("Frekuensi kemunculan (kali)"); ax1.set_ylabel("kata dasar")
            st.pyplot(fig1)
            
        with c_bi:
            st.markdown("**Top 10 Frasa (Bigram)**")
            try:
                vec = CountVectorizer(ngram_range=(2, 2)).fit(df['Clean_Content'])
                bag = vec.transform(df['Clean_Content'])
                sum_words = bag.sum(axis=0) 
                words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)[:10]
                
                df_bigram = pd.DataFrame(words_freq, columns=['Frasa', 'Frekuensi'])
                df_bigram['Frasa'] = df_bigram['Frasa'].apply(lambda x: textwrap.fill(x, 20))
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(data=df_bigram, x='Frekuensi', y='Frasa', hue='Frasa', palette=PALETTE, ax=ax2, legend=False)
                ax2.set_xlabel("Frekuensi kemunculan (kali)"); ax2.set_ylabel("Frasa")
                st.pyplot(fig2)
            except: st.info("Data Bigram belum cukup.")

    with tab3:
        c_vio, c_pie = st.columns(2)
        with c_vio:
            st.markdown("**Statistik Panjang Dokumen (Violin Plot)**")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.violinplot(x=doc_lens, color='#482677', alpha=0.6, ax=ax3, inner="quart", width=0.9, cut=0)
            ax3.set_xlabel("Jumlah Kata per dokumen", fontweight='bold'); ax3.set_yticks([])
            ax3.set_ylabel("Kepadatan Distribusi (Densitas)", fontweight='bold')
            mean_val = doc_lens.mean()
            ax3.axvline(mean_val, color='red', linestyle='--', label=f'Avg: {mean_val:.0f}')
            ax3.legend()
            st.pyplot(fig3)
            
        with c_pie:
            st.markdown("**Sumber Berita (Donut Chart)**")
            if 'Source' in df.columns:
                src_counts = df['Source'].value_counts().head(7)
                colors = sns.color_palette(PALETTE, len(src_counts))
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                ax4.pie(src_counts, labels=src_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85, explode=[0.05]*len(src_counts))
                ax4.add_artist(plt.Circle((0,0), 0.65, fc='white'))
                st.pyplot(fig4)
            else: st.info("Metadata Sumber tidak tersedia.")
       
# --- PAGE 2: PENCARIAN ---
elif menu == "🔍 Mesin Pencari":
    st.markdown("<h2 style='margin-bottom:10px;'>🔍 Pencarian Berita Pendidikan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:25px;'>Cari informasi berita pendidikan dengan algoritma TF-IDF dan BM25</p>", unsafe_allow_html=True)
    
    # Search Bar Container 
    col_in, col_btn = st.columns([4, 1])
    
    with col_in:
        query = st.text_input(
            "Kata Kunci", 
            placeholder="Ketik topik (misal: kurikulum merdeka, beasiswa...)",
            label_visibility="collapsed",
            key="search_query"
        )
    
    with col_btn:
        btn_search = st.button("🔍 TELUSURI", use_container_width=True, type="primary")
            
    if query and btn_search:
        clean_q = preprocess_text(query, stemmer, stop_words)
        
        # Info Query
        st.markdown(f"""
        <div style='background:#f0f9ff; padding:15px; border-radius:10px; border-left:4px solid #3b82f6; margin-bottom:25px;'>
            <b>🔎 Query Asli:</b> <span style='color:#1e40af;'>{query}</span><br>
            <b>✨ Setelah Preprocessing:</b> <code style='background:#dbeafe; padding:3px 8px; border-radius:5px;'>{clean_q}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # TF-IDF
        with st.spinner("⏳ Memproses TF-IDF..."):
            s = time.time()
            q_v = vectorizer.transform([clean_q])
            sc_tf = cosine_similarity(q_v, tfidf_matrix).flatten()
            idx_tf = sc_tf.argsort()[-10:][::-1]
            t_tf = time.time() - s
        
        # BM25
        with st.spinner("⏳ Memproses BM25..."):
            s = time.time()
            sc_bm = bm25.get_scores(clean_q.split())
            idx_bm = np.argsort(sc_bm)[-10:][::-1]
            t_bm = time.time() - s
        
        # Comparison Tabs
        tab_a, tab_b = st.tabs([
            f"🔵 Hasil TF-IDF ({t_tf:.4f} detik)", 
            f"🟡 Hasil BM25 ({t_bm:.4f} detik)"
        ])
        
        def render_results(indices, scores, color_border):
            found = False
            for i in indices:
                if scores[i] > 0.001:
                    found = True
                    row = df.iloc[i]
                    hl_title = highlight_text(row['Title'], query)
                    hl_snip = highlight_text(get_snippet(row['Content'], query), query)
                    
                    st.markdown(f"""
                    <div class="result-container" style="border-left: 5px solid {color_border};">
                        <div class="result-score-badge">Score: {scores[i]:.4f} | Doc ID: {row['Doc_ID']}</div>
                        <div class="result-link"><a href="{row['URL']}" target="_blank">{hl_title}</a></div>
                        <div class="result-meta">
                            <span>📅 {row['Date']}</span>
                            <span>•</span>
                            <span>📰 {row['Source']}</span>
                        </div>
                        <div class="result-snippet">{hl_snip}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if not found: 
                st.warning("⚠️ Tidak ada dokumen yang relevan dengan query Anda. Coba gunakan kata kunci lain.")
            
        with tab_a: 
            render_results(idx_tf, sc_tf, "#2563eb")
            
        with tab_b: 
            render_results(idx_bm, sc_bm, "#eab308")
    
    elif not query and btn_search:
        st.warning("⚠️ Mohon masukkan kata kunci pencarian terlebih dahulu!")
    
    # Info tambahan jika belum search
    if not query or not btn_search:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background:linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                        padding:15px; border-radius:15px; border:2px solid #93c5fd;'>
                <h3 style='color:#1e40af; margin-bottom:5px;'>💡 Tips Pencarian</h3>
                <ul style='color:#1e3a8a; line-height:1.8;'>
                    <li>Gunakan kata kunci spesifik</li>
                    <li>Coba variasi kata yang berbeda</li>
                    <li>Kombinasikan 2-3 kata kunci</li>
                    <li>Contoh: "beasiswa luar negeri"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background:linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        padding:15px; border-radius:15px; border:2px solid #fcd34d;'>
                <h3 style='color:#92400e; margin-bottom:5px;'>🎯 Topik Populer</h3>
                <ul style='color:#78350f; line-height:1.8;'>
                    <li>Kurikulum Merdeka</li>
                    <li>Sekolah Dasar</li>
                    <li>Beasiswa</li>
                    <li>Pendidikan Vokasi</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE 3: EVALUASI ---
elif menu == "⚙️ Evaluasi Kinerja":
    st.markdown("<h2 style='margin-bottom:10px;'>⚙️ Evaluasi Kinerja Sistem Pencarian</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:25px;'>Evaluasi menggunakan Confusion Matrix, Precision, Recall, F1-Score dan analisis Trade-off</p>", unsafe_allow_html=True)
    
    # INFO BOX
    st.info("""
    ℹ️ **Cara Kerja Evaluasi:**
    1. Masukkan query untuk diuji
    2. Sistem akan menampilkan Top-10 hasil dari TF-IDF dan BM25
    3. Anda (sebagai assessor) menandai dokumen mana yang relevan
    4. Sistem menghitung metrik evaluasi berdasarkan pilihan Anda
    """)
    
    # ==========================================
    # INPUT QUERY & SEARCH
    # ==========================================
    st.markdown("### 🔍 Langkah 1: Masukkan Query Evaluasi")
    
    col_q, col_b = st.columns([3, 1])
    with col_q:
        q_eval = st.text_input(
            "Query untuk Evaluasi", 
            placeholder="Contoh: kurikulum merdeka",
            help="Masukkan topik berita pendidikan yang ingin dievaluasi"
        )
    with col_b:
        st.write("")
        st.write("")
        btn_run = st.button("Mulai Evaluasi", use_container_width=True, type="primary")
    
    # Session state untuk menyimpan hasil pencarian
    if 'eval_session' not in st.session_state:
        st.session_state.eval_session = None
    if 'ground_truth_confirmed' not in st.session_state:
        st.session_state.ground_truth_confirmed = False

    # ==========================================
    # EKSEKUSI PENCARIAN
    # ==========================================
    if btn_run and q_eval:
        # Validasi query
        if not q_eval.strip():
            st.warning("⚠️ Query tidak boleh kosong!")
            st.stop()
        
        # Preprocessing
        cln = preprocess_text(q_eval, stemmer, stop_words)
        
        if not cln:
            st.error("❌ Query terlalu pendek atau hanya berisi stopwords! Gunakan kata kunci yang lebih spesifik.")
            st.stop()
        
        # Info Query
        st.markdown(f"""
        <div style='background:#f0f9ff; padding:15px; border-radius:10px; border-left:4px solid #3b82f6; margin:20px 0;'>
            <b>🔎 Query Asli:</b> <span style='color:#1e40af;'>{q_eval}</span><br>
            <b>✨ Setelah Preprocessing:</b> <code style='background:#dbeafe; padding:3px 8px; border-radius:5px;'>{cln}</code>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("⏳ Memproses pencarian dengan TF-IDF dan BM25..."):
            # TF-IDF Search
            q_vec_tfidf = vectorizer.transform([cln])
            scores_tfidf = cosine_similarity(q_vec_tfidf, tfidf_matrix).flatten()
            
            # Ambil top-10 dengan threshold minimum
            MIN_SCORE = 0.001
            valid_tfidf = np.where(scores_tfidf > MIN_SCORE)[0]
            
            if len(valid_tfidf) == 0:
                st.error("❌ TF-IDF tidak menemukan dokumen relevan untuk query ini!")
                st.stop()
            
            idx_tfidf = valid_tfidf[scores_tfidf[valid_tfidf].argsort()[-10:][::-1]]
            top_scores_tfidf = scores_tfidf[idx_tfidf]
            
            # BM25 Search
            scores_bm25 = bm25.get_scores(cln.split())
            valid_bm25 = np.where(scores_bm25 > MIN_SCORE)[0]
            
            if len(valid_bm25) == 0:
                st.error("❌ BM25 tidak menemukan dokumen relevan untuk query ini!")
                st.stop()
            
            idx_bm25 = valid_bm25[scores_bm25[valid_bm25].argsort()[-10:][::-1]]
            top_scores_bm25 = scores_bm25[idx_bm25]
        
        # Simpan ke session state
        st.session_state.eval_session = {
            'query': q_eval,
            'clean_query': cln,
            'idx_tfidf': idx_tfidf,
            'idx_bm25': idx_bm25,
            'scores_tfidf': top_scores_tfidf,
            'scores_bm25': top_scores_bm25
        }
        st.session_state.ground_truth_confirmed = False
        
        st.success("✅ Pencarian selesai! Scroll ke bawah untuk melihat hasil.")

    # ==========================================
    # TAMPILKAN HASIL JIKA ADA SESSION
    # ==========================================
    if st.session_state.eval_session:
        res = st.session_state.eval_session
        
        st.markdown("---")
        
        # ==========================================
        # VISUALISASI SKOR
        # ==========================================
        st.markdown("### 📊 Langkah 2: Perbandingan Skor Ranking")
        
        fig_scores, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        ranks = list(range(1, len(res['scores_tfidf']) + 1))
        
        # Chart TF-IDF
        colors_tfidf = plt.cm.Blues(np.linspace(0.4, 0.9, len(res['scores_tfidf'])))
        bars1 = ax1.barh(ranks, res['scores_tfidf'][::-1], color=colors_tfidf, 
                         edgecolor='navy', linewidth=1.5)
        ax1.set_xlabel('Cosine Similarity Score', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Rank', fontsize=13, fontweight='bold')
        ax1.set_title('🔵 TF-IDF Ranking', fontsize=16, fontweight='bold', pad=20)
        ax1.set_yticks(ranks)
        ax1.set_yticklabels([f'#{i}' for i in reversed(ranks)], fontsize=11)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, score in zip(bars1, res['scores_tfidf'][::-1]):
            ax1.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=11, fontweight='bold')
        
        # Chart BM25
        colors_bm25 = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(res['scores_bm25'])))
        bars2 = ax2.barh(ranks, res['scores_bm25'][::-1], color=colors_bm25, 
                         edgecolor='darkred', linewidth=1.5)
        ax2.set_xlabel('BM25 Score', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Rank', fontsize=13, fontweight='bold')
        ax2.set_title('🟡 BM25 Ranking', fontsize=16, fontweight='bold', pad=20)
        ax2.set_yticks(ranks)
        ax2.set_yticklabels([f'#{i}' for i in reversed(ranks)], fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, score in zip(bars2, res['scores_bm25'][::-1]):
            ax2.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_scores)
        
        st.markdown("---")
        
        # ==========================================
        # GROUND TRUTH SELECTION
        # ==========================================
        st.markdown("### 👤 Langkah 3: Manual Annotation (Relevance Judgment)")
        st.markdown("""
        <div style='background:#fff7ed; padding:15px; border-radius:10px; border-left:4px solid #f59e0b; margin-bottom:20px;'>
            <b>📋 Instruksi:</b> Baca judul dokumen di bawah ini dan centang dokumen yang <b>RELEVAN</b> dengan query <code>{}</code>
        </div>
        """.format(res['query']), unsafe_allow_html=True)
        
        col_tfidf, col_bm25 = st.columns(2)
        
        selected_tfidf = []
        selected_bm25 = []
        
        with col_tfidf:
            st.markdown("#### 🔵 Kandidat dari TF-IDF")
            st.caption("Centang dokumen yang relevan:")
            
            for rank, idx in enumerate(res['idx_tfidf'], 1):
                doc = df.iloc[idx]
                score = res['scores_tfidf'][rank-1]

                is_checked = st.checkbox(
                    f"**Rank #{rank}** — Score: {score:.4f}",
                    key=f"tfidf_{idx}"
                )

                # Tampilkan judul saja
                st.markdown(f"📄 *{doc['Title']}*")

                if is_checked:
                    selected_tfidf.append(idx)

                st.markdown("")  # spasi kecil
                
        with col_bm25:
            st.markdown("#### 🟡 Kandidat dari BM25")
            st.caption("Centang dokumen yang relevan:")
            
            for rank, idx in enumerate(res['idx_bm25'], 1):
                doc = df.iloc[idx]
                score = res['scores_bm25'][rank-1]

                is_checked = st.checkbox(
                    f"**Rank #{rank}** — Score: {score:.4f}",
                    key=f"bm25_{idx}"
                )

                # Tampilkan judul saja
                st.markdown(f"📄 *{doc['Title']}*")

                if is_checked:
                    selected_bm25.append(idx)

                st.markdown("")  # spasi kecil
        
        # Summary seleksi
        ground_truth = set(selected_tfidf).union(set(selected_bm25))
        
        st.markdown(f"""
        <div style='background:#ecfdf5; padding:15px; border-radius:10px; border-left:4px solid #10b981; margin:20px 0;'>
            <b>✅ Anda telah menandai:</b><br>
            • TF-IDF: <b>{len(selected_tfidf)}</b> dokumen relevan<br>
            • BM25: <b>{len(selected_bm25)}</b> dokumen relevan<br>
            • <b>Total dokumen unik yang relevan (Ground Truth): {len(ground_truth)}</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==========================================
        # HITUNG EVALUASI
        # ==========================================
        st.markdown("### 🧮 Langkah 4: Hitung Metrik Evaluasi")
        
        if st.button("📊 HITUNG CONFUSION MATRIX & METRICS", type="primary", use_container_width=True):
            
            if len(ground_truth) == 0:
                st.error("❌ Harap pilih minimal 1 dokumen yang relevan sebagai Ground Truth!")
                st.stop()
            
            st.session_state.ground_truth_confirmed = True
            
            # ==========================================
            # FUNGSI EVALUASI LENGKAP
            # ==========================================
            def calculate_metrics(retrieved, relevant, total_docs):
                """
                Menghitung semua metrik evaluasi IR
                
                Args:
                    retrieved: list of retrieved document indices
                    relevant: set of relevant document indices (ground truth)
                    total_docs: total documents in corpus
                
                Returns:
                    dict dengan semua metrik
                """
                retrieved_set = set(retrieved)
                
                # Confusion Matrix Components
                tp = len(retrieved_set.intersection(relevant))          # True Positive
                fp = len(retrieved) - tp                                 # False Positive
                fn = len(relevant) - tp                                  # False Negative
                tn = total_docs - tp - fp - fn                          # True Negative
                
                # Metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Mean Average Precision (MAP)
                hits = 0
                sum_precisions = 0
                for i, doc_idx in enumerate(retrieved, 1):
                    if doc_idx in relevant:
                        hits += 1
                        sum_precisions += hits / i
                
                average_precision = sum_precisions / len(relevant) if len(relevant) > 0 else 0
                
                return {
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'map': average_precision
                }
            
            # Hitung metrik untuk kedua algoritma
            with st.spinner("⏳ Menghitung metrik evaluasi..."):
                metrics_tfidf = calculate_metrics(
                    retrieved=res['idx_tfidf'].tolist(),
                    relevant=ground_truth,
                    total_docs=len(df)
                )
                
                metrics_bm25 = calculate_metrics(
                    retrieved=res['idx_bm25'].tolist(),
                    relevant=ground_truth,
                    total_docs=len(df)
                )
            
            st.success("✅ Perhitungan selesai!")
            
            # ==========================================
            # TAMPILKAN CONFUSION MATRIX
            # ==========================================
            st.markdown("---")
            st.markdown("### 📋 Confusion Matrix")
            
            col_matrix, col_explain = st.columns([1.2, 1])
            
            with col_matrix:
                # Buat DataFrame untuk heatmap
                matrix_df = pd.DataFrame({
                    'TF-IDF': [
                        metrics_tfidf['tp'],
                        metrics_tfidf['fp'],
                        metrics_tfidf['fn'],
                        metrics_tfidf['tn']
                    ],
                    'BM25': [
                        metrics_bm25['tp'],
                        metrics_bm25['fp'],
                        metrics_bm25['fn'],
                        metrics_bm25['tn']
                    ]
                }, index=['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)'])
                
                # Visualisasi heatmap
                fig_matrix, ax_matrix = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    matrix_df, 
                    annot=True, 
                    fmt='d',
                    cmap='RdYlGn',
                    linewidths=2,
                    linecolor='white',
                    ax=ax_matrix,
                    annot_kws={"size": 16, "weight": "bold"},
                    cbar_kws={'label': 'Jumlah Dokumen'}
                )
                ax_matrix.set_title("Confusion Matrix untuk Information Retrieval", 
                                   fontsize=14, fontweight='bold', pad=20)
                ax_matrix.set_xlabel("Algoritma", fontsize=12, fontweight='bold')
                ax_matrix.set_ylabel("Kategori", fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_matrix)
            
            with col_explain:
                st.markdown("""
                <div style='background:#f0f9ff; padding:20px; border-radius:10px; border:2px solid #3b82f6;'>
                    <h4 style='color:#1e40af; margin-top:0;'>📖 Penjelasan Confusion Matrix</h4>
                    <ul style='color:#1e3a8a; line-height:2;'>
                        <li><b>TP (True Positive):</b><br>Dokumen relevan yang berhasil dikembalikan sistem ✅</li>
                        <li><b>FP (False Positive):</b><br>Dokumen tidak relevan yang dikembalikan sistem ❌</li>
                        <li><b>FN (False Negative):</b><br>Dokumen relevan yang tidak dikembalikan ⚠️</li>
                        <li><b>TN (True Negative):</b><br>Dokumen tidak relevan yang tidak dikembalikan ✓</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # ==========================================
            # TABEL METRIK
            # ==========================================
            st.markdown("---")
            st.markdown("### 📊 Tabel Metrik Evaluasi")
            
            metrics_table = pd.DataFrame({
                "Metrik": ["Precision", "Recall", "F1-Score", "MAP (Mean Avg Precision)"],
                "TF-IDF": [
                    metrics_tfidf['precision'],
                    metrics_tfidf['recall'],
                    metrics_tfidf['f1_score'],
                    metrics_tfidf['map']
                ],
                "BM25": [
                    metrics_bm25['precision'],
                    metrics_bm25['recall'],
                    metrics_bm25['f1_score'],
                    metrics_bm25['map']
                ]
            })
            
            # Tambahkan kolom selisih
            metrics_table['Selisih (TF-IDF - BM25)'] = metrics_table['TF-IDF'] - metrics_table['BM25']
            
            # Format dan styling
            st.dataframe(
                metrics_table.style
                    .format({
                        'TF-IDF': '{:.4f}',
                        'BM25': '{:.4f}',
                        'Selisih (TF-IDF - BM25)': '{:+.4f}'
                    })
                    .background_gradient(subset=['TF-IDF', 'BM25'], cmap='RdYlGn', vmin=0, vmax=1)
                    .set_properties(**{
                        'text-align': 'center',
                        'font-weight': 'bold',
                        'font-size': '14px'
                    })
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#1e293b'), 
                                                     ('color', 'white'), 
                                                     ('font-weight', 'bold'),
                                                     ('text-align', 'center'),
                                                     ('font-size', '15px')]}
                    ]),
                use_container_width=True
            )
            
            # ==========================================
            # BAR CHART PERBANDINGAN
            # ==========================================
            st.markdown("---")
            st.markdown("### 📊 Perbandingan Visual Metrik")
            
            metrics_names = ['Precision', 'Recall', 'F1-Score', 'MAP']
            tfidf_values = [
                metrics_tfidf['precision'],
                metrics_tfidf['recall'],
                metrics_tfidf['f1_score'],
                metrics_tfidf['map']
            ]
            bm25_values = [
                metrics_bm25['precision'],
                metrics_bm25['recall'],
                metrics_bm25['f1_score'],
                metrics_bm25['map']
            ]
            
            fig_comparison, ax_comp = plt.subplots(figsize=(14, 7))
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax_comp.bar(x - width/2, tfidf_values, width, 
                               label='TF-IDF', color='#3b82f6', 
                               edgecolor='black', linewidth=1.5)
            bars2 = ax_comp.bar(x + width/2, bm25_values, width, 
                               label='BM25', color='#f59e0b', 
                               edgecolor='black', linewidth=1.5)
            
            # Tambahkan nilai di atas bar
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', 
                                fontsize=12, fontweight='bold')
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            ax_comp.set_xlabel('Metrik Evaluasi', fontsize=14, fontweight='bold')
            ax_comp.set_ylabel('Nilai Score', fontsize=14, fontweight='bold')
            ax_comp.set_title('Perbandingan Kinerja TF-IDF vs BM25', 
                             fontsize=16, fontweight='bold', pad=20)
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(metrics_names, fontsize=13, fontweight='bold')
            ax_comp.legend(fontsize=13, loc='upper right', framealpha=0.9)
            ax_comp.set_ylim(0, max(max(tfidf_values), max(bm25_values)) * 1.2)
            ax_comp.grid(axis='y', alpha=0.3, linestyle='--')
            ax_comp.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Threshold 0.5')
            
            plt.tight_layout()
            st.pyplot(fig_comparison)
            
            # Penjelasan Trade-off
            st.markdown("---")
            st.markdown("### 🔄 Analisis Trade-off Precision vs Recall")
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        padding: 20px; border-radius: 12px; border-left: 5px solid #0284c7; margin-top: 20px;'>
                <h4 style='color: #0c4a6e; margin-top: 0;'>💡 Memahami Trade-off Precision vs Recall</h4>
                <ul style='color: #0369a1; line-height: 1.8;'>
                    <li><b>Precision tinggi:</b> Sistem hanya mengembalikan dokumen yang sangat yakin relevan → Hasil sedikit tapi akurat</li>
                    <li><b>Recall tinggi:</b> Sistem mengembalikan semua dokumen yang mungkin relevan → Hasil banyak tapi ada noise</li>
                    <li><b>Ideal:</b> Keseimbangan antara keduanya (mendekati garis diagonal Perfect Balance)</li>
                    <li><b>F1-Score:</b> Harmonic mean yang menyeimbangkan Precision dan Recall</li>
                </ul>
                <p style='margin-top: 15px; color: #075985; font-weight: 600;'>
                    📌 Sistem terbaik berada di kuadran kanan atas (Precision > 0.7 DAN Recall > 0.7)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ==========================================
            # ANALISIS TRADE-OFF
            # ==========================================
            col_explain1, col_explain2 = st.columns(2)
            
            with col_explain1:
                # Analisis TF-IDF
                p_tfidf = metrics_tfidf['precision']
                r_tfidf = metrics_tfidf['recall']
                
                if p_tfidf > r_tfidf + 0.15:
                    analysis_tfidf = "⚠️ **Sistem terlalu selektif**<br>Precision tinggi tapi banyak dokumen relevan terlewat (Recall rendah)"
                    color_tfidf = "#fef3c7"
                    border_tfidf = "#f59e0b"
                elif r_tfidf > p_tfidf + 0.15:
                    analysis_tfidf = "⚠️ **Sistem terlalu inklusif**<br>Recall tinggi tapi banyak dokumen tidak relevan masuk (Precision rendah)"
                    color_tfidf = "#fee2e2"
                    border_tfidf = "#ef4444"
                else:
                    analysis_tfidf = "✅ **Sistem seimbang**<br>Trade-off yang baik antara Precision dan Recall"
                    color_tfidf = "#d1fae5"
                    border_tfidf = "#10b981"
                
                st.markdown(f"""
                <div style='background:{color_tfidf}; padding:20px; border-radius:10px; border-left:5px solid {border_tfidf};'>
                    <h4 style='margin-top:0; color:#1e293b;'>🔵 TF-IDF</h4>
                    <ul style='line-height:2; color:#1e293b;'>
                        <li><b>Precision:</b> {p_tfidf:.2%} → {"<span style='color:green'>✓ Tinggi</span>" if p_tfidf > 0.7 else "<span style='color:red'>✗ Rendah</span>"}</li>
                        <li><b>Recall:</b> {r_tfidf:.2%} → {"<span style='color:green'>✓ Tinggi</span>" if r_tfidf > 0.7 else "<span style='color:red'>✗ Rendah</span>"}</li>
                    </ul>
                    <p style='margin:10px 0 0 0; font-weight:bold;'>{analysis_tfidf}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_explain2:
                # Analisis BM25
                p_bm25 = metrics_bm25['precision']
                r_bm25 = metrics_bm25['recall']
                
                if p_bm25 > r_bm25 + 0.15:
                    analysis_bm25 = "⚠️ **Sistem terlalu selektif**<br>Precision tinggi tapi banyak dokumen relevan terlewat (Recall rendah)"
                    color_bm25 = "#fef3c7"
                    border_bm25 = "#f59e0b"
                elif r_bm25 > p_bm25 + 0.15:
                    analysis_bm25 = "⚠️ **Sistem terlalu inklusif**<br>Recall tinggi tapi banyak dokumen tidak relevan masuk (Precision rendah)"
                    color_bm25 = "#fee2e2"
                    border_bm25 = "#ef4444"
                else:
                    analysis_bm25 = "✅ **Sistem seimbang**<br>Trade-off yang baik antara Precision dan Recall"
                    color_bm25 = "#d1fae5"
                    border_bm25 = "#10b981"
                
                st.markdown(f"""
                <div style='background:{color_bm25}; padding:20px; border-radius:10px; border-left:5px solid {border_bm25};'>
                    <h4 style='margin-top:0; color:#1e293b;'>🟡 BM25</h4>
                    <ul style='line-height:2; color:#1e293b;'>
                        <li><b>Precision:</b> {p_bm25:.2%} → {"<span style='color:green'>✓ Tinggi</span>" if p_bm25 > 0.7 else "<span style='color:red'>✗ Rendah</span>"}</li>
                        <li><b>Recall:</b> {r_bm25:.2%} → {"<span style='color:green'>✓ Tinggi</span>" if r_bm25 > 0.7 else "<span style='color:red'>✗ Rendah</span>"}</li>
                    </ul>
                    <p style='margin:10px 0 0 0; font-weight:bold;'>{analysis_bm25}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ==========================================
            # SCATTER PLOT TRADE-OFF
            # ==========================================
            st.markdown("---")
            st.markdown("#### 📍 Visualisasi Trade-off")
            
            fig_tradeoff, ax_trade = plt.subplots(figsize=(12, 8))
            
            # Plot points
            ax_trade.scatter([p_tfidf], [r_tfidf], s=500, c='#3b82f6', 
                            marker='o', label='TF-IDF', 
                            edgecolor='black', linewidth=2.5, zorder=5, alpha=0.8)
            ax_trade.scatter([p_bm25], [r_bm25], s=500, c='#f59e0b', 
                            marker='s', label='BM25', 
                            edgecolor='black', linewidth=2.5, zorder=5, alpha=0.8)
            
            # Annotate points
            ax_trade.annotate(f'TF-IDF\nP={p_tfidf:.3f}\nR={r_tfidf:.3f}', 
                             xy=(p_tfidf, r_tfidf), 
                             xytext=(p_tfidf+0.08, r_tfidf-0.08),
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', 
                                      facecolor='#dbeafe', 
                                      edgecolor='#3b82f6',
                                      linewidth=2),
                             arrowprops=dict(arrowstyle='->', 
                                           connectionstyle='arc3,rad=0.3',
                                           color='#3b82f6', 
                                           lw=2))
            
            ax_trade.annotate(f'BM25\nP={p_bm25:.3f}\nR={r_bm25:.3f}', 
                             xy=(p_bm25, r_bm25), 
                             xytext=(p_bm25-0.15, r_bm25+0.08),
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', 
                                      facecolor='#fef3c7', 
                                      edgecolor='#f59e0b',
                                      linewidth=2),
                             arrowprops=dict(arrowstyle='->', 
                                           connectionstyle='arc3,rad=-0.3',
                                           color='#f59e0b', 
                                           lw=2))
            
            # Ideal line
            ax_trade.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Balance (P=R)')
            
            # High performance zones
            ax_trade.fill_between([0.7, 1], 0, 1, alpha=0.05, color='green')
            ax_trade.axhline(y=0.7, color='green', linestyle=':', alpha=0.3, linewidth=1.5)
            ax_trade.axvline(x=0.7, color='blue', linestyle=':', alpha=0.3, linewidth=1.5)
            ax_trade.text(0.85, 0.05, 'High Precision\nZone', fontsize=9, 
                         color='blue', alpha=0.6, ha='center', fontweight='bold')
            ax_trade.text(0.05, 0.85, 'High Recall\nZone', fontsize=9, 
                         color='green', alpha=0.6, ha='center', fontweight='bold')
            
            ax_trade.set_xlabel('Precision', fontsize=13, fontweight='bold')
            ax_trade.set_ylabel('Recall', fontsize=13, fontweight='bold')
            ax_trade.set_title('Trade-off Precision vs Recall Analysis', 
                              fontsize=15, fontweight='bold', pad=20)
            ax_trade.legend(fontsize=11, loc='lower left', framealpha=0.9)
            ax_trade.grid(alpha=0.3, linestyle='--')
            ax_trade.set_xlim(-0.05, 1.1)
            ax_trade.set_ylim(-0.05, 1.1)
            
            plt.tight_layout()
            st.pyplot(fig_tradeoff)
            
            # Plot points
            ax_trade.scatter([p_tfidf], [r_tfidf], s=500, c='#3b82f6', 
                            marker='o', label='TF-IDF', 
                            edgecolor='black', linewidth=2.5, zorder=5, alpha=0.8)
            ax_trade.scatter([p_bm25], [r_bm25], s=500, c='#f59e0b', 
                            marker='s', label='BM25', 
                            edgecolor='black', linewidth=2.5, zorder=5, alpha=0.8)
            
            col_winner, col_stats = st.columns([1, 1])
            
            with col_winner:
                # Tentukan pemenang berdasarkan F1-Score
                if metrics_tfidf['f1_score'] > metrics_bm25['f1_score']:
                    winner = "TF-IDF"
                    winner_color = "#3b82f6"
                    winner_emoji = "🔵"
                elif metrics_bm25['f1_score'] > metrics_tfidf['f1_score']:
                    winner = "BM25"
                    winner_color = "#f59e0b"
                    winner_emoji = "🟡"
                else:
                    winner = "SEIMBANG"
                    winner_color = "#10b981"
                    winner_emoji = "⚖️"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {winner_color}22 0%, {winner_color}11 100%); 
                            padding: 25px; border-radius: 15px; border: 3px solid {winner_color}; text-align: center;'>
                    <h2 style='color: {winner_color}; margin: 0; font-size: 48px;'>{winner_emoji}</h2>
                    <h3 style='color: #1e293b; margin: 10px 0;'>Algoritma Terbaik</h3>
                    <h1 style='color: {winner_color}; margin: 5px 0; font-size: 36px;'>{winner}</h1>
                    <p style='color: #64748b; margin-top: 10px; font-size: 14px;'>
                        Berdasarkan F1-Score untuk query ini
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats:
                st.markdown("""
                <div style='background: #f8fafc; padding: 20px; border-radius: 12px; border: 2px solid #e2e8f0;'>
                    <h4 style='color: #1e293b; margin-top: 0;'>📊 Ringkasan Performa</h4>
                """, unsafe_allow_html=True)
                
                # Comparison table
                comparison_data = {
                    "": ["TF-IDF", "BM25"],
                    "TP": [metrics_tfidf['tp'], metrics_bm25['tp']],
                    "FP": [metrics_tfidf['fp'], metrics_bm25['fp']],
                    "Precision": [f"{metrics_tfidf['precision']:.3f}", f"{metrics_bm25['precision']:.3f}"],
                    "Recall": [f"{metrics_tfidf['recall']:.3f}", f"{metrics_bm25['recall']:.3f}"],
                    "F1": [f"{metrics_tfidf['f1_score']:.3f}", f"{metrics_bm25['f1_score']:.3f}"]
                }
                
                st.dataframe(
                    pd.DataFrame(comparison_data).set_index(""),
                    use_container_width=True,
                    height=120
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Rekomendasi
            st.markdown("---")
            st.markdown("### 💡 Rekomendasi")
            
            avg_f1 = (metrics_tfidf['f1_score'] + metrics_bm25['f1_score']) / 2
            
            if avg_f1 > 0.7:
                recommendation = "✅ **Performa Excellent** - Kedua algoritma memberikan hasil yang sangat baik untuk query ini."
                rec_color = "#d1fae5"
                rec_border = "#10b981"
            elif avg_f1 > 0.5:
                recommendation = "⚠️ **Performa Good** - Hasil cukup baik, namun masih ada ruang untuk improvement dengan tuning parameter."
                rec_color = "#fef3c7"
                rec_border = "#f59e0b"
            else:
                recommendation = "❌ **Performa Poor** - Perlu evaluasi ulang query atau pertimbangkan preprocessing yang lebih baik."
                rec_color = "#fee2e2"
                rec_border = "#ef4444"
            
            st.markdown(f"""
            <div style='background: {rec_color}; padding: 20px; border-radius: 12px; border-left: 5px solid {rec_border};'>
                <p style='color: #1e293b; margin: 0; font-size: 16px; line-height: 1.8;'>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE 4: DATASET ---
elif menu == "📂 Dataset Korpus":
    st.markdown("<h2 style='margin-bottom:10px;'>📂 Eksplorasi Dataset</h2>", unsafe_allow_html=True)
    
    view_mode = st.radio("Mode Tampilan:", ["Tabel Lengkap", "Bandingkan Raw vs Clean"], horizontal=True)
    
    if view_mode == "Tabel Lengkap":
        st.dataframe(df, use_container_width=True, height=1000)
    else:
        idx = st.number_input("Pilih Index Dokumen", 0, len(df)-1, 0)
        c1, c2 = st.columns(2)
        with c1: 
            st.info("📄 Dokumen Asli (Raw)")
            st.text_area("Content", df.iloc[idx]['Content'], height=400, disabled=True)
        with c2: 
            st.success("✨ Hasil Stemming (Clean)")
            st.text_area("Clean", df.iloc[idx]['Clean_Content'], height=400, disabled=True)
            
# --- PAGE 5: TENTANG ---
elif menu == "ℹ️ Tentang":
    st.markdown("<h2>ℹ️ Tentang Aplikasi</h2>", unsafe_allow_html=True)

    st.markdown("""
    EduNews Search adalah mesin pencari vertikal yang dirancang khusus untuk 
    mengindeks dan melakukan pencarian pada berita pendidikan Indonesia.

    **Fitur Utama:**
    - Algoritma TF-IDF dan BM25  
    - Analisis statistik korpus  
    - Evaluasi kinerja retrieval  
    - Korpus berita pendidikan Indonesia  

    **Dikembangkan oleh Kelompok 10:**
    - Dea Zasqia Pasaribu Malau  (2308107100004)
    - Tasya Zahrani             (2308107010006)
    - Adinda Muarriva       (2308107010001)

    Proyek ini dibuat untuk memenuhi tugas Mata Kuliah **Pencarian Informasi**,  
    Informatika, Universitas Syiah Kuala.
    """)