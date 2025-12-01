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
# 1. KONFIGURASI HALAMAN & CSS SUPER PREMIUM
# ==========================================
st.set_page_config(
    page_title="EduSearch Pro - Sistem Temu Kembali Informasi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download Resource NLTK (Silent Mode)
try: nltk.data.find('corpora/stopwords')
except LookupError: nltk.download('stopwords')

# --- CUSTOM CSS (TAMPILAN MEWAH & PROFESIONAL) ---
st.markdown("""
<style>
    /* 1. IMPORT FONT PREMIUM (Poppins & Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b; /* Dark Slate for Text */
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #0f172a !important;
    }

    /* 2. SIDEBAR GELAP (NAVY THEME) - TEKS PUTIH */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    /* Memaksa semua elemen di sidebar jadi putih */
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important; 
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 8px;
        transition: background 0.3s;
    }
    /* Efek Hover Menu Sidebar */
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255,255,255,0.1);
        cursor: pointer;
    }
    /* Menu Aktif */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }

    /* 3. CARD METRIK (DASHBOARD) */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 6px;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .metric-val { 
        font-size: 38px; 
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
        margin-top: 5px;
    }

    /* 4. TOMBOL VISUALISASI (TABS) YANG LEBIH BESAR */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 65px; /* Tinggi Tab Diperbesar */
        background-color: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 16px; /* Font Besar */
        font-weight: 600;
        flex: 1; /* Tab memenuhi lebar */
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        color: #2563eb;
    }
    /* Tab Aktif (Selected) */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }

    /* 5. HASIL PENCARIAN (RESULT BOX) */
    .result-box {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: all 0.3s;
    }
    .result-box:hover {
        box-shadow: 0 15px 30px rgba(0,0,0,0.08);
        transform: translateY(-3px);
        border-color: #bfdbfe;
    }
    .result-title a {
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: #1e40af;
        text-decoration: none;
    }
    .result-title a:hover { text-decoration: underline; color: #1e3a8a; }
    .result-meta {
        font-size: 13px;
        color: #64748b;
        margin: 8px 0 12px 0;
        display: flex;
        gap: 15px;
        align-items: center;
    }
    .score-badge {
        background-color: #eff6ff;
        color: #2563eb;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        border: 1px solid #dbeafe;
    }
    .result-snippet {
        font-size: 15px;
        color: #334155;
        line-height: 1.6;
    }

    /* 6. TOMBOL UTAMA (GRADIENT BUTTON) */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #06b6d4 100%);
        color: white;
        border-radius: 10px;
        border: none;
        height: 50px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 15px rgba(37, 99, 235, 0.25);
        color: white;
    }

    /* 7. HIGHLIGHT & UTILS */
    .highlight { 
        background-color: #fef08a; 
        color: #854d0e; 
        padding: 2px 4px; 
        border-radius: 4px; 
        font-weight: 700; 
    }
    
    /* Input Field Styling */
    div[data-baseweb="input"] {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        padding: 5px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI UTAMA (BACKEND LOGIC)
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
    """Smart Loader: Membaca data yang sudah diproses di notebook jika ada."""
    file_nb = 'Lampiran_Data_Bersih.csv'
    file_meta = 'Lampiran_Data_Mentah.csv'
    
    # 1. Cek File Cache Notebook (Prioritas)
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

    # 2. Fallback: Load Raw & Process
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
        
        # Preprocessing on the fly (Tampilkan semua data tanpa limit)
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

# --- HELPER GUI ---
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
# 3. INITIALIZATION
# ==========================================
with st.spinner("🚀 Memuat Data & AI Engine..."):
    df, status_msg = load_dataset()
    if df.empty:
        st.error(f"Gagal memuat data. Pesan: {status_msg}")
        st.stop()
    df['Clean_Content'] = df['Clean_Content'].fillna('').astype(str)
    vectorizer, tfidf_matrix, bm25 = build_engine(df['Clean_Content'].tolist())
    stemmer, stop_words = get_resources()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f1/Universitas_Syiah_Kuala_logo.svg", width=90)
    st.markdown("<h2 style='color: white; margin-top: 10px;'>EduSearch Pro</h2>", unsafe_allow_html=True)
    st.caption(f"Status: {status_msg}")
    st.markdown("---")
    
    menu = st.radio("MENU NAVIGASI", 
                    ["Dashboard", "Mesin Pencari", "Evaluasi Kinerja", "Dataset Korpus"],
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(f"**📚 Total Dokumen:** {len(df)}")
    st.markdown("<br><div style='color:#94a3b8; font-size:12px;'>© 2025 Kelompok 10<br>Teknologi Pencarian Informasi</div>", unsafe_allow_html=True)

# ==========================================
# 4. HALAMAN UTAMA (KONTEN)
# ==========================================

# --- PAGE 1: DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2 style='margin-bottom:10px;'>📊 Dashboard Analisis Korpus</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:30px;'>Statistik visual mendalam mengenai distribusi kata dan topik dalam korpus berita pendidikan.</p>", unsafe_allow_html=True)
    
    # Metrics
    all_text = " ".join(df['Clean_Content'])
    doc_lens = df['Clean_Content'].apply(lambda x: len(x.split()))
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-lbl">Total Artikel</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-card"><div class="metric-val">{int(doc_lens.mean())}</div><div class="metric-lbl">Rata-rata Kata</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(set(all_text.split())):,}</div><div class="metric-lbl">Kosakata Unik</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # VISUALISASI UTAMA (TABS BESAR)
    tab1, tab2, tab3 = st.tabs(["☁️ WordCloud Besar", "📊 Tren Kata & Frasa", "📉 Distribusi Data"])
    
    sns.set_style("whitegrid")
    PALETTE = "viridis"
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
 
    with tab1:
        st.subheader("Kata Kunci Paling Dominan")
        # Wordcloud HD
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
                # FIX BIGRAM VISUALIZATION LOGIC
                vec = CountVectorizer(ngram_range=(2, 2)).fit(df['Clean_Content'])
                bag = vec.transform(df['Clean_Content'])
                sum_words = bag.sum(axis=0) 
                words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)[:10]
                
                # Buat DataFrame khusus untuk Seaborn agar rapi
                df_bigram = pd.DataFrame(words_freq, columns=['Frasa', 'Frekuensi'])
                
                # Wrap text agar tidak menabrak
                df_bigram['Frasa'] = df_bigram['Frasa'].apply(lambda x: textwrap.fill(x, 20))
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                
                # Plot menggunakan DataFrame (lebih stabil)
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
elif menu == "Mesin Pencari":
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
elif menu == "Evaluasi Kinerja":
    st.markdown("<h2 style='margin-bottom:10px;'>⚙️ Evaluasi Kinerja (Matrix & Metrics)</h2>", unsafe_allow_html=True)
    
    st.info("ℹ️ **Cara Kerja:** Karena dataset ini unsupervised, Anda berperan sebagai pakar (Ground Truth) untuk menandai dokumen relevan.")
    
    c_q, c_b = st.columns([3, 1])
    with c_q: q_eval = st.text_input("Query Uji", "pendidikan vokasi")
    with c_b: 
        st.write("")
        st.write("")
        btn_run = st.button("Mulai Evaluasi")
        
    if 'eval_session' not in st.session_state: st.session_state.eval_session = None
    
    if btn_run:
        cln = preprocess_text(q_eval, stemmer, stop_words)
        idx_t = cosine_similarity(vectorizer.transform([cln]), tfidf_matrix).flatten().argsort()[-10:][::-1]
        idx_b = np.argsort(bm25.get_scores(cln.split()))[-10:][::-1]
        st.session_state.eval_session = {'tf': idx_t, 'bm': idx_b}
        
    if st.session_state.eval_session:
        res = st.session_state.eval_session
        
        st.markdown("### 👉 Tandai Dokumen Relevan")
        c1, c2 = st.columns(2)
        sel_t, sel_b = [], []
        
        with c1:
            st.subheader("Kandidat TF-IDF")
            for i, idx in enumerate(res['tf']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"t_{i}"): sel_t.append(idx)
        with c2:
            st.subheader("Kandidat BM25")
            for i, idx in enumerate(res['bm']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"b_{i}"): sel_b.append(idx)
        
        st.markdown("---")
        if st.button("🧮 HITUNG MATRIX EVALUASI"):
            truth = set(sel_t).union(set(sel_b))
            if not truth: st.error("Harap pilih minimal 1 dokumen yang relevan!"); st.stop()
            
            # Helper
            def get_stats(retrieved, relevant):
                ret_set = set(retrieved)
                tp = len(ret_set.intersection(relevant))
                fp = len(retrieved) - tp
                prec = tp/len(retrieved); rec = tp/len(relevant)
                f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0
                hits=0; sum_p=0
                for i, x in enumerate(retrieved):
                    if x in relevant: hits+=1; sum_p+=hits/(i+1)
                ap = sum_p/len(relevant)
                return tp, fp, prec, rec, f1, ap
            
            tp1, fp1, p1, r1, f11, map1 = get_stats(res['tf'], truth)
            tp2, fp2, p2, r2, f12, map2 = get_stats(res['bm'], truth)
            
            # 1. MATRIX HEATMAP
            st.subheader("🔥 Retrieval Performance Matrix")
            col_mat, col_tab = st.columns([1, 1])
            
            with col_mat:
                matrix_data = pd.DataFrame({'TF-IDF': [tp1, fp1], 'BM25': [tp2, fp2]}, index=['Relevan (TP)', 'Irrelevan (FP)'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues', linewidths=1, ax=ax, annot_kws={"size": 16, "weight": "bold"})
                ax.set_title("Confusion Matrix (Top-10 Results)")
                st.pyplot(fig)
            
            # 2. METRIC TABLE
            with col_tab:
                eval_df = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1-Score", "MAP"],
                    "TF-IDF": [p1, r1, f11, map1],
                    "BM25": [p2, r2, f12, map2]
                }).set_index("Metric")
                st.table(eval_df.style.format("{:.4f}").background_gradient(cmap="Greens", axis=1))
            
            # 3. KESIMPULAN
            winner = "BM25" if map2 > map1 else "TF-IDF"
            st.success(f"🏆 **Analisis:** Berdasarkan skor MAP, algoritma **{winner}** memberikan hasil yang lebih relevan untuk query ini.")

# --- PAGE 4: DATASET ---
elif menu == "Dataset Korpus":
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