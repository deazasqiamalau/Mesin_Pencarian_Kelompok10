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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# --- 1. KONFIGURASI HALAMAN (HARUS PERTAMA) ---
st.set_page_config(
    page_title="EduSearch - Sistem Temu Kembali Informasi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DOWNLOAD RESOURCE NLTK (SILENT) ---
try: nltk.data.find('corpora/stopwords')
except LookupError: nltk.download('stopwords')
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 3. CUSTOM CSS (PROFESSIONAL UI/UX) ---
st.markdown("""
<style>
    /* Import Font Keren */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    /* SIDEBAR PROFESSIONAL STYLE */
    [data-testid="stSidebar"] {
        background-color: #0f172a; /* Dark Navy */
        color: #f8fafc;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #94a3b8;
    }
    
    /* Tombol Utama */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: scale(1.02);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    /* Kartu Metrik Dashboard */
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1e293b;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Result Cards */
    .result-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    .result-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    .highlight {
        background-color: #fef08a; /* Yellow highlight */
        padding: 0 2px;
        border-radius: 2px;
        font-weight: 600;
        color: #854d0e;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. FUNGSI LOAD DATA (ROBUST) ---
@st.cache_data
def load_and_fix_csv(filename):
    """Membaca CSV dengan penanganan error kolom yang kuat"""
    cleaned_rows = []
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try: header = next(reader)
            except StopIteration: return pd.DataFrame()
            
            for row in reader:
                while row and row[-1] == '': row.pop()
                if not row: continue
                
                # Gunakan 'http' sebagai anchor point untuk memperbaiki kolom geser
                url_index = -1
                for i, col in enumerate(row):
                    if 'http' in col:
                        url_index = i
                        break
                
                if url_index != -1:
                    doc_id = row[0]
                    title = ",".join(row[1 : url_index - 1]).strip().strip('"')
                    source = row[url_index - 1].strip()
                    url = row[url_index].strip()
                    publish_date = ",".join(row[url_index + 1 : -1]).strip().strip('"')
                    content = row[-1].strip().strip('"')
                    cleaned_rows.append([doc_id, title, source, url, publish_date, content])
                    
        return pd.DataFrame(cleaned_rows, columns=['Doc_ID', 'Title', 'Source', 'URL', 'Date', 'Content'])
    except FileNotFoundError: return pd.DataFrame()

# --- 5. PREPROCESSING & MODELING (CACHED) ---
@st.cache_resource
def get_resources():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    # Tambah stopword berita
    custom_stop = {'baca', 'juga', 'halaman', 'kompas', 'detik', 'com', 'wib', 'jakarta', 'copyright', 
                   'advertisement', 'shutterstock', 'photo', 'news', 'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu',
                   'untuk', 'pada', 'adalah', 'sebagai', 'dengan', 'dalam', 'tribun', 'liputan6', 'penulis', 'editor'}
    stop_words.update(custom_stop)
    return stemmer, stop_words

def preprocess_text(text, stemmer, stop_words):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # Hapus angka & simbol
    tokens = text.split()
    clean_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 3]
    return " ".join(clean_tokens)

@st.cache_data
def load_dataset():
    """Smart Loader: Cek cache notebook dulu, kalau gagal load raw"""
    file_nb = 'Lampiran_Data_Bersih.csv'
    file_meta = 'Lampiran_Data_Mentah.csv'
    
    # 1. Coba load dari hasil Notebook (Cepat)
    if os.path.exists(file_nb) and os.path.exists(file_meta):
        try:
            df_nb = pd.read_csv(file_nb)
            # FORCE LOAD RAW jika data notebook cuma sampel (< 100 baris)
            if len(df_nb) > 100: 
                df_meta = pd.read_csv(file_meta)
                df = pd.merge(df_meta, df_nb[['Doc_ID', 'Clean_Content']], on='Doc_ID', how='left')
                df['Clean_Content'] = df['Clean_Content'].fillna('')
                return df, "Cache Notebook (Cepat)"
        except: pass
    
    # 2. Fallback: Load Raw & Process (Lama)
    df_raw = load_and_fix_csv('korpus_pendidikan_gabungan.csv')
    if df_raw.empty: return pd.DataFrame(), "Error"
    
    stemmer, stop_words = get_resources()
    
    # Tampilkan loading state
    placeholder = st.empty()
    bar = st.progress(0)
    clean_content = []
    
    for i, row in df_raw.iterrows():
        clean_content.append(preprocess_text(row['Content'], stemmer, stop_words))
        if i % 10 == 0:
            prog = (i+1)/len(df_raw)
            bar.progress(prog)
            placeholder.text(f"⚙️ Sedang melakukan Stemming data {i+1}/{len(df_raw)}... Mohon tunggu.")
            
    df_raw['Clean_Content'] = clean_content
    placeholder.empty()
    bar.empty()
    return df_raw, "Raw Processed"

@st.cache_resource
def build_search_engine(corpus):
    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # BM25
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return vectorizer, tfidf_matrix, bm25

# --- HELPER GUI ---
def highlight_text(text, query):
    if not query: return text
    query_terms = query.lower().split()
    query_terms.sort(key=len, reverse=True)
    for term in query_terms:
        if len(term) > 2:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<span class='highlight'>{m.group(0)}</span>", text)
    return text

def get_snippet(text, query, limit=160):
    if not query: return text[:limit] + "..."
    text_lower = text.lower()
    start = -1
    for q in query.lower().split():
        start = text_lower.find(q)
        if start != -1: break
    
    if start == -1: return text[:limit] + "..."
    
    start_pos = max(0, start - 60)
    end_pos = min(len(text), start + 100)
    return ("..." if start_pos>0 else "") + text[start_pos:end_pos] + ("..." if end_pos<len(text) else "")

# --- 6. INISIALISASI ---
with st.spinner("🚀 Memuat Sistem EduSearch..."):
    df, status = load_dataset()
    if df.empty:
        st.error("Data korpus tidak ditemukan. Pastikan file CSV ada.")
        st.stop()
    
    vectorizer, tfidf_matrix, bm25 = build_search_engine(df['Clean_Content'].tolist())
    stemmer, stop_words = get_resources()

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("🎓 EduSearch")
    st.caption("Penelusuran Informasi Pendidikan")
    st.markdown("---")
    
    # MENU UTAMA (Updated)
    menu = st.radio("MENU UTAMA", 
        ["Dashboard", "Pencarian (Search)", "Visualisasi Korpus", "📂 Dataset Korpus", "Evaluasi Sistem"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Status Sistem")
    st.markdown(f"**Dokumen:** {len(df)}")
    st.markdown(f"**Mode Data:** {status}")
    st.markdown(f"**Algoritma:** TF-IDF & BM25")
    
    st.markdown("---")
    st.info("**Kelompok 10**\n\nAdinda Muarriva\nDea Zasqia P. Malau\nTasya Zahrani")

# ================= MENU 1: DASHBOARD (HOME) =================
if menu == "Dashboard":
    st.header("Selamat Datang di EduSearch 🎓")
    st.markdown("Sistem temu kembali informasi (Information Retrieval) khusus untuk topik **Pendidikan di Indonesia**.")
    
    # 3 Kartu Metrik Utama
    c1, c2, c3 = st.columns(3)
    
    all_text = " ".join(df['Clean_Content'])
    total_words = len(all_text.split())
    vocab_size = len(set(all_text.split()))
    
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Dokumen</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_words:,}</div>
            <div class="metric-label">Total Kata (Token)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{vocab_size:,}</div>
            <div class="metric-label">Kosakata Unik</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibangun untuk memenuhi **Projek Akhir Mata Kuliah Penelusuran Informasi 2025**.
    
    **Fitur Utama:**
    * 🔍 **Vertical Search Engine:** Fokus pada domain pendidikan.
    * 🤖 **Dual Algorithm:** Membandingkan TF-IDF (Vector Space Model) dan BM25 (Probabilistic Model).
    * 📈 **Visual Analytics:** Analisis korpus menggunakan WordCloud dan N-Grams.
    * ⚙️ **Evaluation Metrics:** Perhitungan otomatis Precision, Recall, F1-Score, dan MAP.
    """)

# ================= MENU 2: PENCARIAN (SEARCH) =================
elif menu == "Pencarian (Search)":
    st.title("🔍 Pencarian Berita")
    
    # Search Bar Mewah
    col_search, col_act = st.columns([5, 1])
    with col_search:
        query = st.text_input("", placeholder="Ketik kata kunci (misal: kurikulum merdeka, beasiswa)...", label_visibility="collapsed")
    with col_act:
        search = st.button("CARI")
        
    if query:
        # Preprocessing Query
        clean_q = preprocess_text(query, stemmer, stop_words)
        
        # 1. TF-IDF Process
        start = time.time()
        q_vec = vectorizer.transform([clean_q])
        tfidf_sc = cosine_similarity(q_vec, tfidf_matrix).flatten()
        idx_tfidf = tfidf_sc.argsort()[-10:][::-1]
        time_tfidf = time.time() - start
        
        # 2. BM25 Process
        start = time.time()
        bm25_sc = bm25.get_scores(clean_q.split())
        idx_bm25 = np.argsort(bm25_sc)[-10:][::-1]
        time_bm25 = time.time() - start
        
        st.success(f"Ditemukan hasil untuk: **'{query}'** (Processed: '{clean_q}')")
        
        # TABS HASIL
        tab_t, tab_b = st.tabs(["🔵 Hasil TF-IDF", "🟡 Hasil BM25"])
        
        # Fungsi Render Hasil
        def render_results(indices, scores, time_taken, method_name, color_border):
            st.caption(f"⏱️ Waktu Eksekusi: {time_taken:.5f} detik")
            if scores[indices[0]] == 0:
                st.warning("Tidak ada dokumen yang relevan ditemukan.")
                return

            for rank, i in enumerate(indices):
                score = scores[i]
                if score > 0.001:
                    row = df.iloc[i]
                    snippet = highlight_text(get_snippet(row['Content'], query), query)
                    title = highlight_text(row['Title'], query)
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 5px solid {color_border};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0;"><a href="{row['URL']}" target="_blank" style="text-decoration:none; color:#1e293b;">{title}</a></h4>
                            <span style="background-color:{color_border}; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">Score: {score:.4f}</span>
                        </div>
                        <div style="font-size:12px; color:#64748b; margin-top:5px;">
                            📅 {row['Date']} | 📰 {row['Source']} | 🏅 Rank #{rank+1}
                        </div>
                        <p style="font-size:14px; color:#334155; margin-top:10px; line-height:1.5;">
                            {snippet}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_t:
            render_results(idx_tfidf, tfidf_sc, time_tfidf, "TF-IDF", "#2563eb") # Blue
            
        with tab_b:
            render_results(idx_bm25, bm25_sc, time_bm25, "BM25", "#f59e0b") # Amber

# ================= MENU 3: VISUALISASI PRO =================
elif menu == "Visualisasi Korpus":
    st.title("📈 Visualisasi Data Korpus")
    
    # Setup Data Visual
    all_text = " ".join(df['Clean_Content'])
    words = all_text.split()
    word_counts = Counter(words)
    
    # Gunakan Seaborn Theme
    sns.set_theme(style="whitegrid")
    
    tab1, tab2, tab3 = st.tabs(["Kata Kunci (WordCloud)", "Frekuensi Kata (Bar)", "Distribusi Data"])
    
    with tab1:
        st.subheader("☁️ WordCloud")
        st.caption("Representasi visual kata-kata yang paling sering muncul dalam korpus.")
        
        wc = WordCloud(
            width=1200, height=600, 
            background_color='white', 
            colormap='cividis', # Warna pro (biru-kuning)
            max_words=150,
            contour_width=0
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
    with tab2:
        st.subheader("📊 Top 15 Kata Kunci")
        
        df_freq = pd.DataFrame(word_counts.most_common(15), columns=['Kata', 'Frekuensi'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_freq, x='Frekuensi', y='Kata', palette='viridis', ax=ax)
        ax.set_title("15 Kata Paling Dominan", fontsize=14, fontweight='bold')
        ax.set_xlabel("Jumlah Kemunculan")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
    with tab3:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**📄 Distribusi Panjang Dokumen**")
            doc_lens = df['Clean_Content'].apply(lambda x: len(x.split()))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(doc_lens, kde=True, color='#4e73df', bins=20, ax=ax)
            ax.set_xlabel("Jumlah Kata")
            st.pyplot(fig)
            
        with c2:
            st.markdown("**📰 Sumber Berita**")
            if 'Source' in df.columns:
                src_counts = df['Source'].value_counts().head(5)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = sns.color_palette('pastel')[0:5]
                ax.pie(src_counts, labels=src_counts.index, autopct='%.1f%%', colors=colors, startangle=90)
                st.pyplot(fig)

# ================= MENU 4: DATASET KORPUS (BARU) =================
elif menu == "📂 Dataset Korpus":
    st.title("📂 Eksplorasi Dataset")
    st.markdown(f"Menampilkan seluruh **{len(df)} dokumen** yang tersimpan dalam sistem.")
    
    # Pilihan Tampilan
    view_option = st.radio("Pilih Tampilan:", ["Tabel Lengkap", "Perbandingan Raw vs Clean"], horizontal=True)
    
    if view_option == "Tabel Lengkap":
        st.dataframe(df, use_container_width=True, height=600)
    else:
        # Tampilan Compare
        doc_idx = st.number_input("Masukkan Index Dokumen (0 - {})".format(len(df)-1), min_value=0, max_value=len(df)-1, value=0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("📄 Dokumen Asli (Raw)")
            st.text_area("Content", df.iloc[doc_idx]['Content'], height=400, disabled=True)
        with c2:
            st.success("✨ Hasil Stemming (Clean)")
            st.text_area("Clean_Content", df.iloc[doc_idx]['Clean_Content'], height=400, disabled=True)

# ================= MENU 5: EVALUASI LENGKAP =================
elif menu == "Evaluasi Sistem":
    st.title("⚙️ Evaluasi Kinerja (Metrics)")
    
    st.markdown("""
    <div style="background-color:#e0f2fe; padding:15px; border-radius:8px; border-left:5px solid #0ea5e9; margin-bottom:20px;">
        <strong>Metode Evaluasi: Relevance Feedback</strong><br>
        Karena tidak ada <i>Ground Truth</i> (kunci jawaban) mutlak, sistem menggunakan penilaian manual pengguna.
        Centang dokumen yang menurut Anda <b>RELEVAN</b> dengan query untuk menghitung skor.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Query Evaluasi
    col_ev, _ = st.columns([2,1])
    with col_ev:
        q_eval = st.text_input("Masukkan Query Evaluasi", value="pendidikan vokasi")
        btn_eval = st.button("Tampilkan Dokumen untuk Dinilai")
        
    # State Management
    if 'eval_data' not in st.session_state:
        st.session_state.eval_data = None

    if btn_eval:
        clean_q = preprocess_text(q_eval, stemmer, stop_words)
        
        # Retrieve TF-IDF
        v_vec = vectorizer.transform([clean_q])
        s_tf = cosine_similarity(v_vec, tfidf_matrix).flatten()
        idx_tf = s_tf.argsort()[-10:][::-1]
        
        # Retrieve BM25
        s_bm = bm25.get_scores(clean_q.split())
        idx_bm = np.argsort(s_bm)[-10:][::-1]
        
        st.session_state.eval_data = {'tfidf': idx_tf, 'bm25': idx_bm}

    # Tampilan Checklist
    if st.session_state.eval_data:
        res = st.session_state.eval_data
        
        st.markdown("---")
        st.write("Silakan centang dokumen yang **Relevan**:")
        
        col_t, col_b = st.columns(2)
        
        sel_t, sel_b = [], []
        
        with col_t:
            st.subheader("Hasil TF-IDF")
            for i, idx in enumerate(res['tfidf']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"tf_{i}"):
                    sel_t.append(idx)
                    
        with col_b:
            st.subheader("Hasil BM25")
            for i, idx in enumerate(res['bm25']):
                if st.checkbox(f"{i+1}. {df.iloc[idx]['Title']}", key=f"bm_{i}"):
                    sel_b.append(idx)
        
        st.markdown("---")
        
        if st.button("📊 HITUNG PRECISION, RECALL, F1, MAP"):
            # Ground Truth = Gabungan unik dokumen relevan yang dipilih user
            ground_truth = set(sel_t).union(set(sel_b))
            
            if not ground_truth:
                st.error("Pilih minimal 1 dokumen yang relevan!")
            else:
                # Helper Hitung Metrik
                def calc_metrics(retrieved, truth):
                    ret_set = set(retrieved)
                    tp = len(ret_set.intersection(truth))
                    
                    precision = tp / len(retrieved)
                    recall = tp / len(truth)
                    f1 = (2 * precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
                    
                    # MAP Calculation
                    hits = 0
                    sum_p = 0
                    for i, idx in enumerate(retrieved):
                        if idx in truth:
                            hits += 1
                            sum_p += hits / (i+1)
                    ap = sum_p / len(truth) if truth else 0
                    
                    return precision, recall, f1, ap

                p1, r1, f1_1, map1 = calc_metrics(res['tfidf'], ground_truth)
                p2, r2, f1_2, map2 = calc_metrics(res['bm25'], ground_truth)
                
                # Tampilkan Hasil dalam Tabel Cantik
                st.success("Perhitungan Selesai!")
                
                score_data = {
                    "Metrik": ["Precision@10", "Recall", "F1-Score", "MAP (Mean Avg Precision)"],
                    "TF-IDF": [f"{p1:.4f}", f"{r1:.4f}", f"{f1_1:.4f}", f"{map1:.4f}"],
                    "BM25": [f"{p2:.4f}", f"{r2:.4f}", f"{f1_2:.4f}", f"{map2:.4f}"]
                }
                
                st.table(pd.DataFrame(score_data).set_index("Metrik"))
                
                # Rumus Penjelasan
                with st.expander("📚 Penjelasan Rumus Metrik"):
                    st.latex(r"Precision = \frac{|Relevant \cap Retrieved|}{|Retrieved|}")
                    st.latex(r"Recall = \frac{|Relevant \cap Retrieved|}{|Total Relevant|}")
                    st.latex(r"F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}")
                    st.latex(r"MAP = \frac{1}{|Q|} \sum_{q \in Q} AveragePrecision(q)")