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
# 1. KONFIGURASI HALAMAN & CSS PROFESIONAL
# ==========================================
st.set_page_config(
    page_title="EduSearch - Sistem Temu Kembali Informasi",
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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 1. Sidebar Styling (Navy Blue Theme) */
    [data-testid="stSidebar"] {
        background-color: #0f172a; /* Warna Navy Gelap */
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 15px;
        padding: 10px;
        border-radius: 5px;
        transition: background 0.3s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #1e293b;
        color: white !important;
    }
    
    /* 2. Main Area Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2563eb; /* Aksen Biru */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-val { font-size: 28px; font-weight: 800; color: #1e293b; }
    .metric-lbl { font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }
    
    /* 3. Result Box (Kartu Hasil Pencarian) */
    .result-box {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        position: relative;
    }
    .result-box:hover {
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
        top: -2px;
    }
    .result-title a {
        text-decoration: none;
        color: #0f172a;
        font-weight: 700;
        font-size: 18px;
        transition: color 0.2s;
    }
    .result-title a:hover { color: #2563eb; }
    .result-meta {
        font-size: 12px;
        color: #64748b;
        margin-top: 5px;
        margin-bottom: 10px;
        display: flex;
        gap: 15px;
    }
    .result-score {
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 11px;
    }
    .result-snippet {
        font-size: 14px;
        color: #334155;
        line-height: 1.6;
    }
    
    /* 4. Custom Buttons */
    .stButton>button {
        background: linear-gradient(to right, #2563eb, #1d4ed8);
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: 700;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1d4ed8, #1e40af);
        box-shadow: 0 6px 10px rgba(37, 99, 235, 0.3);
    }
    
    /* 5. Highlight Marker */
    .highlight {
        background-color: #fef08a;
        padding: 0 3px;
        border-radius: 3px;
        color: #854d0e;
        font-weight: 700;
        border-bottom: 2px solid #eab308;
    }
    
    /* 6. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px;
        background-color: #f1f5f9;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
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
    """
    Load Data Cerdas: Prioritaskan file hasil olahan Notebook (.csv bersih) 
    agar aplikasi tidak perlu melakukan stemming ulang (instan).
    """
    file_nb = 'Lampiran_Data_Bersih.csv'
    file_meta = 'Lampiran_Data_Mentah.csv'
    
    # 1. Cek File Cache Notebook
    if os.path.exists(file_nb) and os.path.exists(file_meta):
        try:
            df_nb = pd.read_csv(file_nb)
            if len(df_nb) > 50: 
                df_meta = pd.read_csv(file_meta)
                # Join Metadata dengan Clean Content
                if 'Doc_ID' in df_meta.columns and 'Doc_ID' in df_nb.columns:
                    df = pd.merge(df_meta, df_nb[['Doc_ID', 'Clean_Content']], on='Doc_ID', how='left')
                    df['Clean_Content'] = df['Clean_Content'].fillna('')
                    return df, "Cache Notebook (Ready)"
        except Exception: pass

    # 2. Fallback: Load Raw & Process (Jika file notebook hilang)
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
                    doc_id = row[0]
                    title = ",".join(row[1 : url_idx - 1]).strip().strip('"')
                    source = row[url_idx - 1].strip()
                    url = row[url_idx].strip()
                    date = ",".join(row[url_idx + 1 : -1]).strip().strip('"')
                    content = row[-1].strip().strip('"')
                    cleaned_rows.append([doc_id, title, source, url, date, content])
                    
        df_raw = pd.DataFrame(cleaned_rows, columns=['Doc_ID', 'Title', 'Source', 'URL', 'Date', 'Content'])
        
        # Preprocessing on the fly (Tanpa batasan limit agar semua data masuk)
        stemmer, stop_words = get_resources()
        # REVISI: Menghapus batasan .head(200) agar semua data termuat
        df_raw['Clean_Content'] = df_raw['Content'].apply(lambda x: preprocess_text(str(x), stemmer, stop_words))
        
        return df_raw, "Mode: Full Raw Processing"
        
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
# 3. INITIALIZATION & SIDEBAR
# ==========================================

# Load Data
with st.spinner("🚀 Memuat Sistem Cerdas..."):
    df, status_msg = load_dataset()
    if df.empty:
        st.error(f"Gagal memuat data. Pesan: {status_msg}")
        st.stop()
    df['Clean_Content'] = df['Clean_Content'].fillna('').astype(str)
    vectorizer, tfidf_matrix, bm25 = build_engine(df['Clean_Content'].tolist())
    stemmer, stop_words = get_resources()

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white;'>🎓 EduSearch</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Sistem Temu Kembali Informasi</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Menu Navigasi
    menu = st.radio("NAVIGASI UTAMA", 
                    ["Dashboard", "Mesin Pencari", "Evaluasi Kinerja", "Dataset"],
                    label_visibility="collapsed")
    
    st.markdown("---")
    
    # Info Dataset
    st.markdown("### 📂 Info Data")
    c1, c2 = st.columns(2)
    with c1: st.metric("Dokumen", len(df))
    with c2: st.metric("Status", "Ready")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748b; font-size: 12px;'>© Kelompok 10<br>Penelusuran Informasi 2025</div>", unsafe_allow_html=True)

# ==========================================
# 4. HALAMAN UTAMA (KONTEN)
# ==========================================

# --- PAGE 1: DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2 style='color:#1e293b;'>📊 Dashboard Analisis Korpus</h2>", unsafe_allow_html=True)
    st.markdown("Analisis visual mendalam mengenai distribusi kata dan topik dalam korpus berita pendidikan.")
    
    # Metrics Overview
    all_text = " ".join(df['Clean_Content'])
    doc_lens = df['Clean_Content'].apply(lambda x: len(x.split()))
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-lbl">Total Artikel</div></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div class="metric-card"><div class="metric-val">{int(doc_lens.mean())}</div><div class="metric-lbl">Rata-rata Kata</div></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div class="metric-card"><div class="metric-val">{len(set(all_text.split())):,}</div><div class="metric-lbl">Kosakata Unik</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisasi Utama
    tab1, tab2, tab3 = st.tabs(["☁️ WordCloud", "📈 Tren Kata", "📉 Distribusi"])
    
    sns.set_style("whitegrid")
    PALETTE = "viridis"
    
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
            ax1.set_xlabel("Frekuensi"); ax1.set_ylabel("")
            st.pyplot(fig1)
            
        with c_bi:
            st.markdown("**Top 10 Frasa (Bigram)**")
            try:
                vec = CountVectorizer(ngram_range=(2, 2)).fit(df['Clean_Content'])
                bag = vec.transform(df['Clean_Content'])
                sum_words = bag.sum(axis=0) 
                words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()], key=lambda x: x[1], reverse=True)[:10]
                
                x_bi, y_bi = zip(*words_freq)
                labels_wrap = [textwrap.fill(lbl, 20) for lbl in x_bi]
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(x=list(y_bi), y=list(range(len(y_bi))), hue=list(x_bi), palette=PALETTE, ax=ax2, legend=False)
                ax2.set_yticks(range(len(y_bi))); ax2.set_yticklabels(labels_wrap)
                ax2.set_xlabel("Frekuensi"); ax2.set_ylabel("")
                st.pyplot(fig2)
            except: st.info("Data Bigram belum cukup.")

    with tab3:
        c_vio, c_pie = st.columns(2)
        with c_vio:
            st.markdown("**Statistik Panjang Dokumen (Violin Plot)**")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.violinplot(x=doc_lens, color='#482677', alpha=0.6, ax=ax3, inner="quart", width=0.9, cut=0)
            ax3.set_xlabel("Jumlah Kata", fontweight='bold'); ax3.set_yticks([])
            ax3.set_ylabel("Densitas", fontweight='bold')
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
    st.markdown("<h2 style='color:#1e293b;'>🔍 Pencarian Berita Pendidikan</h2>", unsafe_allow_html=True)
    
    # Search Box Design
    with st.container():
        st.markdown("<div style='margin-bottom: 10px;'>Masukkan kata kunci pencarian:</div>", unsafe_allow_html=True)
        c_in, c_go = st.columns([5, 1])
        with c_in:
            query = st.text_input("Search", placeholder="Contoh: kurikulum merdeka, beasiswa...", label_visibility="collapsed")
        with c_go:
            do_search = st.button("TELUSURI")
            
    if query:
        # Process
        clean_q = preprocess_text(query, stemmer, stop_words)
        
        # TF-IDF
        s = time.time()
        q_vec = vectorizer.transform([clean_q])
        sc_tf = cosine_similarity(q_vec, tfidf_matrix).flatten()
        idx_tf = sc_tf.argsort()[-10:][::-1]
        t_tf = time.time() - s
        
        # BM25
        s = time.time()
        sc_bm = bm25.get_scores(clean_q.split())
        idx_bm = np.argsort(sc_bm)[-10:][::-1]
        t_bm = time.time() - s
        
        st.markdown(f"**Hasil untuk:** `{query}` (Processed: `{clean_q}`)")
        
        # Comparison Tabs
        tab_a, tab_b = st.tabs([f"🔵 TF-IDF ({t_tf:.4f}s)", f"🟡 BM25 ({t_bm:.4f}s)"])
        
        def render_results(indices, scores, color_border):
            found = False
            for i in indices:
                if scores[i] > 0.001:
                    found = True
                    row = df.iloc[i]
                    hl_title = highlight_text(row['Title'], query)
                    hl_snip = highlight_text(get_snippet(row['Content'], query), query)
                    
                    st.markdown(f"""
                    <div class="result-box" style="border-left: 5px solid {color_border}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="result-title"><a href="{row['URL']}" target="_blank">{hl_title}</a></div>
                            <span class="result-score">Score: {scores[i]:.4f}</span>
                        </div>
                        <div class="result-meta">
                            <span>📅 {row['Date']}</span> &nbsp;•&nbsp; <span>📰 {row['Source']}</span>
                        </div>
                        <div class="result-snippet">{hl_snip}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not found: st.warning("Tidak ada dokumen yang relevan.")
            
        with tab_a: render_results(idx_tf, sc_tf, "#2563eb")
        with tab_b: render_results(idx_bm, sc_bm, "#f59e0b")

# --- PAGE 3: EVALUASI ---
elif menu == "Evaluasi Kinerja":
    st.markdown("<h2 style='color:#1e293b;'>⚙️ Evaluasi Kinerja (Relevance Feedback)</h2>", unsafe_allow_html=True)
    
    st.info("Karena dataset ini **unsupervised**, evaluasi dilakukan dengan metode **User Relevance Feedback**. Anda bertindak sebagai pakar yang menilai relevansi dokumen.")
    
    col_q, col_b = st.columns([3, 1])
    with col_q:
        q_eval = st.text_input("Query Uji", "pendidikan vokasi")
    with col_b:
        st.write("")
        st.write("") # Spacer
        btn_run = st.button("Mulai Evaluasi")
        
    if 'eval_session' not in st.session_state: st.session_state.eval_session = None
    
    if btn_run:
        cln = preprocess_text(q_eval, stemmer, stop_words)
        # Get Candidates
        tf_sc = cosine_similarity(vectorizer.transform([cln]), tfidf_matrix).flatten()
        idx_t = tf_sc.argsort()[-10:][::-1]
        
        bm_sc = bm25.get_scores(cln.split())
        idx_b = np.argsort(bm_sc)[-10:][::-1]
        
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
        if st.button("📊 HITUNG METRIK EVALUASI"):
            truth = set(sel_t).union(set(sel_b))
            if not truth: st.error("Harap pilih minimal 1 dokumen yang relevan!"); st.stop()
            
            # 1. Calculation Logic
            def calculate_detailed(retrieved, relevant):
                ret_set = set(retrieved)
                tp = len(ret_set.intersection(relevant))
                fp = len(retrieved) - tp
                
                prec = tp / len(retrieved) if retrieved.size > 0 else 0
                rec = tp / len(relevant) if len(relevant) > 0 else 0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                # MAP
                hits = 0
                sum_p = 0
                for i, x in enumerate(retrieved):
                    if x in relevant:
                        hits += 1
                        sum_p += hits / (i + 1)
                ap = sum_p / len(relevant) if len(relevant) > 0 else 0
                
                return prec, rec, f1, ap, tp, fp

            p1, r1, f11, map1, tp1, fp1 = calculate_detailed(res['tf'], truth)
            p2, r2, f12, map2, tp2, fp2 = calculate_detailed(res['bm'], truth)
            
            # 2. Display Result Table
            st.success("Perhitungan Selesai!")
            
            res_df = pd.DataFrame({
                "Metric": ["Precision@10", "Recall", "F1-Score", "MAP"],
                "TF-IDF": [p1, r1, f11, map1],
                "BM25": [p2, r2, f12, map2]
            }).set_index("Metric")
            
            st.table(res_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=1))
            
            # 3. Chart Perbandingan
            st.subheader("Grafik Perbandingan Algoritma")
            df_melt = res_df.reset_index().melt(id_vars="Metric", var_name="Algoritma", value_name="Skor")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df_melt, x="Metric", y="Skor", hue="Algoritma", palette="viridis", ax=ax)
            ax.set_ylim(0, 1.1); ax.legend(loc='upper right')
            st.pyplot(fig)

            # 4. Detail Perhitungan (Rumus)
            st.markdown("### 📝 Rincian Perhitungan Manual")
            st.info("Berikut adalah detail perhitungan berdasarkan dokumen yang Anda tandai sebagai relevan.")
            
            tab_tf, tab_bm = st.tabs(["📘 Detail TF-IDF", "📙 Detail BM25"])
            
            total_rel = len(truth)
            
            with tab_tf:
                st.markdown(f"**Data:** Diambil 10 dokumen. Relevan (TP) = {tp1}. Total Ground Truth = {total_rel}")
                st.latex(rf"Precision = \frac{{TP}}{{Retrieved}} = \frac{{{tp1}}}{{10}} = {p1:.4f}")
                st.latex(rf"Recall = \frac{{TP}}{{Total Relevan}} = \frac{{{tp1}}}{{{total_rel}}} = {r1:.4f}")
                st.latex(rf"F1 = 2 \times \frac{{Precision \times Recall}}{{Precision + Recall}} = {f11:.4f}")
                st.latex(rf"MAP = \frac{{\sum P@k}}{{Total Relevan}} = {map1:.4f}")

            with tab_bm:
                st.markdown(f"**Data:** Diambil 10 dokumen. Relevan (TP) = {tp2}. Total Ground Truth = {total_rel}")
                st.latex(rf"Precision = \frac{{TP}}{{Retrieved}} = \frac{{{tp2}}}{{10}} = {p2:.4f}")
                st.latex(rf"Recall = \frac{{TP}}{{Total Relevan}} = \frac{{{tp2}}}{{{total_rel}}} = {r2:.4f}")
                st.latex(rf"F1 = 2 \times \frac{{Precision \times Recall}}{{Precision + Recall}} = {f12:.4f}")
                st.latex(rf"MAP = \frac{{\sum P@k}}{{Total Relevan}} = {map2:.4f}")

# --- PAGE 4: DATASET ---
elif menu == "Dataset":
    st.markdown("<h2 style='color:#1e293b;'>📂 Eksplorasi Dataset</h2>", unsafe_allow_html=True)
    
    view_mode = st.radio("Pilih Tampilan:", ["Tabel Data Lengkap", "Perbandingan Raw vs Clean"], horizontal=True)
    
    if view_mode == "Tabel Data Lengkap":
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