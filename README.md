# MESIN PENCARI VERTIKAL UNTUK BERITA PENDIDIKAN BERBAHASA INDONESIA MENGGUNAKAN TF-IDF DAN BM25

**(Kelompok 10)**

---

## 1. Deskripsi Proyek

Proyek ini merupakan implementasi **Mesin Pencarian (Search Engine)** untuk artikel-artikel bertema pendidikan yang bersumber dari portal berita **Kompas Edu**.

Mesin pencarian dikembangkan menggunakan pendekatan **Vector Space Model (VSM)** dengan pembobotan **TF-IDF (Term Frequency–Inverse Document Frequency)** serta perhitungan **Cosine Similarity** untuk menentukan tingkat relevansi antara *query* (kata kunci pencarian) dan dokumen dalam korpus.

Aplikasi pencarian disajikan dalam bentuk **antarmuka web interaktif menggunakan Streamlit**, sehingga pengguna dapat melakukan pencarian artikel pendidikan secara langsung melalui browser.

---

## 2. Fitur Utama

* **Web Crawling**
  Mengambil data artikel pendidikan dari portal Kompas Edu.

* **Preprocessing Teks**
  Tahapan pengolahan teks meliputi:

  * *Case Folding*
  * Penghapusan tanda baca (*Punctuation Removal*)
  * *Tokenization*
  * Penghapusan *Stopword* menggunakan **NLTK**
  * *Stemming* Bahasa Indonesia menggunakan **Sastrawi**

* **Pembangunan Indeks Terbalik (Inverted Index)**
  Digunakan untuk mempercepat proses pencarian dokumen.

* **Sistem Peringkat Dokumen**
  Menggunakan perhitungan **TF-IDF** dan **Cosine Similarity** untuk mengurutkan dokumen berdasarkan tingkat relevansi terhadap *query*.

* **Antarmuka Web Berbasis Streamlit**
  Menyediakan tampilan pencarian yang sederhana dan interaktif.

---

## 3. Instalasi dan Menjalankan Program

### 3.1 Prasyarat

* Python 3.x
* pip (Python package manager)

### 3.2 Instalasi Dependensi

Instal seluruh dependensi yang dibutuhkan:

```bash
pip install streamlit pandas numpy nltk Sastrawi
```

Unduh resource NLTK (jika belum tersedia):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### 3.3 Menjalankan Aplikasi

1. Pastikan file data berikut tersedia:

   * `Lampiran_Data_Bersih.csv`
   * `korpus_pendidikan_gabungan.csv`
   * Folder `korpus_pendidikan/`

2. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

3. Aplikasi akan terbuka otomatis di browser pada alamat:

```
http://localhost:8501
```

---

## 4. Struktur Folder dan File

```
Mesin_Pencarian_Kelompok10/
│
├── foto/
│   ├── dea.jpg
│   ├── dinda.jpg
│   └── tasya.jpg
│
├── korpus_pendidikan/
│
├── .gitignore
├── app.py
├── crawled_data.py
├── kompasEdu.py
├── korpus_pendidikan_gabungan.csv
├── Lampiran_Data_Mentah.csv
├── Lampiran_Data_Bersih.csv
└── uas_fix.ipynb
```

### Keterangan File

| File / Folder                    | Deskripsi                                                              |
| -------------------------------- | ---------------------------------------------------------------------- |
| `app.py`                         | Aplikasi utama Streamlit dan logika mesin pencarian.                   |
| `crawled_data.py`                | Skrip pengolahan data hasil crawling artikel.                          |
| `kompasEdu.py`                   | Skrip Python untuk melakukan crawling artikel Kompas Edu.              |
| `uas_fix.ipynb`                  | Notebook Jupyter berisi proses preprocessing, inverted index, dan VSM. |
| `Lampiran_Data_Mentah.csv`       | Dataset mentah hasil crawling.                                         |
| `Lampiran_Data_Bersih.csv`       | Dataset hasil pembersihan data.                                        |
| `korpus_pendidikan_gabungan.csv` | Dataset gabungan korpus artikel pendidikan.                            |
| `korpus_pendidikan/`             | Folder penyimpanan dokumen korpus.                                     |
| `foto/`                          | Folder berisi foto anggota kelompok.                                   |

---

## 5. Tim Pengembang

Proyek ini dikembangkan oleh **Kelompok 10**, yang terdiri dari:

* **Adinda Muarriva (2308107010001)**
* **Dea Zasqia Pasaribu Malau (2308107010004)**
* **Tasya Zahrani (2308107010006)**

