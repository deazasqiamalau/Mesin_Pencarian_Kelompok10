import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import time
import os
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
    initial_sidebar_state="collapsed"
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
        margin-bottom: 20px;
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
        margin: 50px 0;
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
        margin-bottom: 40px;
        font-family: 'Poppins', sans-serif;
    }
    
    .team-member {
        text-align: center;
        padding: 30px 20px;
        transition: transform 0.3s;
    }
    
    .team-member:hover {
        transform: translateY(-10px);
    }
    
    .team-photo {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s;
    }
    
    .team-member:hover .team-photo {
        border-color: #764ba2;
        box-shadow: 0 15px 40px rgba(118, 75, 162, 0.4);
        transform: scale(1.05);
    }
    
    .team-name {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
        font-family: 'Poppins', sans-serif;
    }
    
    .team-role {
        font-size: 14px;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
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

# ==========================================
# 2. FUNGSI BACKEND (SAMA SEPERTI SEBELUMNYA)
# ==========================================

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
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Team Section
    st.markdown("""
    <div class="team-section">
        <div class="team-title">👥 Tim Pengembang</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_team1, col_team2, col_team3 = st.columns(3)
    
    with col_team1:
        st.markdown("""
        <div class="team-member">
            <img src="dea.jpg" class="team-photo" onerror="this.src='https://via.placeholder.com/180/667eea/ffffff?text=DEA'">
            <div class="team-name">Dea Zasqia Pasaribu Malau</div>
            <div class="team-role">2308107010004</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_team2:
        st.markdown("""
        <div class="team-member">
            <img src="tasya.jpg" class="team-photo" onerror="this.src='https://via.placeholder.com/180/764ba2/ffffff?text=TASYA'">
            <div class="team-name">Tasya Zahrani</div>
            <div class="team-role">2308107010006</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_team3:
        st.markdown("""
        <div class="team-member">
            <img src="dinda.jpg" class="team-photo" onerror="this.src='https://via.placeholder.com/180/667eea/ffffff?text=DIAN'">
            <div class="team-name">Adinda Muarriva</div>
            <div class="team-role">2308107010001</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
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
            st.markdown(f"""
            <div class="stats-bar">
                <b>📈 Preview Dataset:</b> {len(df_preview)} Dokumen Siap Dianalisis | 
                <b>🏢 Sumber:</b> {df_preview['Source'].nunique() if 'Source' in df_preview.columns else 'N/A'} Media | 
                <b>🗓️</b> Data Terkini
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
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

    menu = st.radio(
        "___",
        ["🏠 Dashboard", "🔍 Mesin Pencari", "⚙️ Evaluasi Kinerja", "📂 Dataset Korpus", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#334155; margin-top:25px;'>", unsafe_allow_html=True)

    st.markdown("""
        <p style="color:#475569; font-size:12px; text-align:center;">
            © 2025 Kelompok 10<br>Teknologi Pencarian Informasi
        </p>
    """, unsafe_allow_html=True)

    # Statistics
    st.markdown(f"""
    <div class="sidebar-stat">
        <div style='font-size:11px; color:#a0aec0; margin-bottom:4px;'>TOTAL DOKUMEN</div>
        <div style='font-size:22px; font-weight:700;'>📄 {len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Source' in df.columns:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div style='font-size:11px; color:#a0aec0; margin-bottom:4px;'>SUMBER MEDIA</div>
            <div style='font-size:22px; font-weight:700;'>🏢 {df['Source'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin:25px 0;'></div>", unsafe_allow_html=True)
    
    # Back to Home Button
    if st.button("🏠 Kembali ke Beranda", use_container_width=True):
        st.session_state.app_started = False
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style='position:fixed; bottom:20px; left:20px; right:20px; text-align:center; 
                font-size:11px; color:#64748b; padding-top:15px; border-top:1px solid rgba(255,255,255,0.1);'>
        © 2025 Kelompok 10<br>
        <span style='color:#667eea;'>Teknologi Pencarian Informasi</span>
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
    
    # Search Bar Container
    with st.container():
        st.markdown("<div style='background:#f8fafc; padding:20px; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:20px;'>", unsafe_allow_html=True)
        c_in, c_go = st.columns([5, 1])
        with c_in:
            query = st.text_input("Kata Kunci", placeholder="Ketik topik (misal: kurikulum merdeka, beasiswa...)", label_visibility="collapsed")
        with c_go:
            st.write("") # Spacer agar sejajar
            btn_search = st.button("TELUSURI")
        st.markdown("</div>", unsafe_allow_html=True)
            
    if query:
        clean_q = preprocess_text(query, stemmer, stop_words)
        st.markdown(f"**Query Processed:** `{clean_q}`")
        
        # TF-IDF
        s = time.time()
        q_v = vectorizer.transform([clean_q])
        sc_tf = cosine_similarity(q_v, tfidf_matrix).flatten()
        idx_tf = sc_tf.argsort()[-10:][::-1]
        t_tf = time.time() - s
        
        # BM25
        s = time.time()
        sc_bm = bm25.get_scores(clean_q.split())
        idx_bm = np.argsort(sc_bm)[-10:][::-1]
        t_bm = time.time() - s
        
        # Comparison Tabs
        tab_a, tab_b = st.tabs([f"🔵 Hasil TF-IDF ({t_tf:.4f}s)", f"🟡 Hasil BM25 ({t_bm:.4f}s)"])
        
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
                        <div class="result-score-badge">Score: {scores[i]:.4f}</div>
                        <div class="result-link"><a href="{row['URL']}" target="_blank">{hl_title}</a></div>
                        <div class="result-meta">
                            <span>📅 {row['Date']}</span>
                            <span>•</span>
                            <span>📰 {row['Source']}</span>
                        </div>
                        <div class="result-snippet">{hl_snip}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not found: st.warning("Tidak ada dokumen yang relevan.")
            
        with tab_a: render_results(idx_tf, sc_tf, "#2563eb")
        with tab_b: render_results(idx_bm, sc_bm, "#eab308")

# --- PAGE 3: EVALUASI ---
elif menu == "⚙️ Evaluasi Kinerja":
    st.markdown("<h2 style='margin-bottom:10px;'>⚙️ Evaluasi Kinerja (Matrix & Metrics)</h2>", unsafe_allow_html=True)
    
    st.info("ℹ️ Karena dataset ini unsupervised, Anda sebagai pengguna berperan memberi penandaan dokumen relevan (Ground Truth).")
    
    # Input Query
    col_q, col_b = st.columns([3, 1])
    with col_q:
        q_eval = st.text_input("Query Uji", "pendidikan vokasi")
    with col_b:
        st.write(""); st.write("")
        btn_run = st.button("Mulai Evaluasi")
    
    # Session state
    if 'eval_session' not in st.session_state:
        st.session_state.eval_session = None

    # Jalankan Sistem Pencari untuk Top-10
    if btn_run:
        cln = preprocess_text(q_eval, stemmer, stop_words)
        
        # TF-IDF ranking
        idx_t = cosine_similarity(vectorizer.transform([cln]), tfidf_matrix).flatten()\
                   .argsort()[-10:][::-1]
        
        # BM25 ranking
        idx_b = np.argsort(bm25.get_scores(cln.split()))[-10:][::-1]
        
        # Simpan sesi
        st.session_state.eval_session = {'tf': idx_t, 'bm': idx_b}
    
    # Jika sesi ranking ada
    if st.session_state.eval_session:
        res = st.session_state.eval_session
        
        st.markdown("### 👉 Tandai Dokumen Relevan")
        col_tf, col_bm = st.columns(2)
        sel_t, sel_b = [], []
        
        with col_tf:
            st.subheader("Kandidat TF-IDF")
            for i, idx in enumerate(res['tf']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"t_{i}"):
                    sel_t.append(idx)

        with col_bm:
            st.subheader("Kandidat BM25")
            for i, idx in enumerate(res['bm']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"b_{i}"):
                    sel_b.append(idx)
        
        st.markdown("---")
        
        # Tombol Hitung Evaluasi
        if st.button("🧮 HITUNG MATRIX EVALUASI"):

            truth = set(sel_t).union(set(sel_b))
            if not truth:
                st.error("Harap pilih minimal 1 dokumen yang relevan!")
                st.stop()

            # ===========================
            #  Fungsi evaluasi
            # ===========================
            def get_stats(retrieved, relevant):
                ret_set = set(retrieved)
                tp = len(ret_set.intersection(relevant))
                fp = len(retrieved) - tp

                prec = tp / len(retrieved)
                rec = tp / len(relevant)
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

                # Hitung MAP/AP
                hits, sum_p = 0, 0
                for i, x in enumerate(retrieved):
                    if x in relevant:
                        hits += 1
                        sum_p += hits / (i + 1)
                ap = sum_p / len(relevant)

                return tp, fp, prec, rec, f1, ap

            # Hitung evaluasi TF-IDF & BM25
            tp1, fp1, p1, r1, f11, map1 = get_stats(res['tf'], truth)
            tp2, fp2, p2, r2, f12, map2 = get_stats(res['bm'], truth)

            # ===========================
            #  1. HEATMAP MATRIX
            # ===========================
            st.subheader("🔥 Retrieval Performance Matrix")
            col_mat, col_tab = st.columns([1, 1])

            with col_mat:
                matrix_data = pd.DataFrame({
                    'TF-IDF': [tp1, fp1],
                    'BM25': [tp2, fp2]
                }, index=['Relevan (TP)', 'Irrelevan (FP)'])

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(matrix_data, annot=True, fmt='d',
                            cmap='Blues', linewidths=1,
                            ax=ax, annot_kws={"size": 16, "weight": "bold"})
                ax.set_title("Confusion Matrix (Top-10 Results)")
                st.pyplot(fig)

            # ===========================
            #  2. TABEL METRIK
            # ===========================
            with col_tab:
                eval_df = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1-Score", "MAP"],
                    "TF-IDF": [p1, r1, f11, map1],
                    "BM25": [p2, r2, f12, map2]
                }).set_index("Metric")

                st.table(
                    eval_df.style
                        .format("{:.4f}")
                        .background_gradient(cmap="Greens", axis=1)
                )

            # ===========================
            #  3. KESIMPULAN AKHIR
            # ===========================
            if map1 == map2:
                st.info("📌 Analisis: Kedua algoritma (TF-IDF dan BM25) memberikan performa **yang sama** untuk query ini.")
            elif map1 > map2:
                st.success("🏆 Analisis: Berdasarkan skor MAP, algoritma **TF-IDF** memberikan hasil yang lebih relevan.")
            else:
                st.success("🏆 Analisis: Berdasarkan skor MAP, algoritma **BM25** memberikan hasil yang lebih relevan.")


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