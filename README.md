# MESIN PENCARI VERTIKAL UNTUK BERITA PENDIDIKAN BERBAHASA INDONESIA MENGGUNAKAN TF-IDF DAN BM25

## 1. Deskripsi Proyek

Proyek ini merupakan implementasi **Mesin Pencarian (Search Engine)** untuk artikel-artikel bertema pendidikan yang bersumber dari portal berita **Kompas Edu**.

Mesin pencarian dikembangkan menggunakan pendekatan **Vector Space Model (VSM)** dengan pembobotan **TF-IDF (Term Frequency–Inverse Document Frequency)** serta perhitungan **Cosine Similarity** untuk menentukan tingkat relevansi antara *query* (kata kunci pencarian) dan dokumen dalam korpus.
Aplikasi dilengkapi dengan antarmuka berbasis web yang dibangun menggunakan *framework* **Flask**.

---

## 2. Fitur Utama

* **Web Crawler**
  Mengambil data artikel pendidikan dari sumber daring (Kompas Edu).

* **Preprocessing Teks**
  Meliputi:

  * *Case Folding*
  * Penghapusan tanda baca (*Punctuation Removal*)
  * *Tokenization*
  * Penghapusan *Stopword* menggunakan **NLTK**
  * *Stemming* Bahasa Indonesia menggunakan **Sastrawi**

* **Indeks Terbalik (Inverted Index)**
  Digunakan untuk mempercepat proses pencarian dokumen.

* **Sistem Peringkat (Ranking)**
  Menggunakan perhitungan **TF-IDF** dan **Cosine Similarity** untuk menampilkan hasil pencarian berdasarkan tingkat relevansi.

* **Antarmuka Web**
  Antarmuka pencarian interaktif berbasis web menggunakan **Flask**.

---

## 3. Instalasi

### 3.1 Prasyarat

Pastikan **Python 3.x** dan **pip** telah terinstal pada sistem Anda.

### 3.2 Instalasi Dependensi

Seluruh dependensi dapat diinstal melalui `pip` dengan perintah berikut:

```bash
# Instal Sastrawi untuk stemming Bahasa Indonesia
pip install Sastrawi

# Instal Flask, Pandas, dan NumPy
pip install Flask pandas numpy

# Instal NLTK untuk stopword removal
pip install nltk

# Unduh resource NLTK (jika belum tersedia)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3.3 Menjalankan Aplikasi

1. **Pastikan Data Tersedia**
   Pastikan file berikut berada pada direktori yang sesuai dan dapat diakses oleh `app.py`:

   * `Lampiran_Data_Bersih.csv`
   * `inverted_index.json` (dihasilkan dari `uas_fix.ipynb`)

2. **Jalankan Server Flask**

   ```bash
   python app.py
   ```

3. **Akses Aplikasi**
   Buka browser dan akses alamat berikut:

   ```
   http://127.0.0.1:5000/
   ```

---

## 4. Struktur Folder dan File

| Nama File / Folder         | Deskripsi                                                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                   | Logika utama aplikasi Flask dan fungsi mesin pencarian (preprocessing, VSM, dan ranking).                                    |
| `uas_fix.ipynb`            | Notebook Jupyter berisi proses pengembangan: pembersihan data, preprocessing, pembuatan inverted index, dan perhitungan VSM. |
| `kompasEdu.py`             | Skrip Python untuk melakukan crawling artikel Kompas Edu.                                                                    |
| `Lampiran_Data_Mentah.csv` | Dataset mentah hasil crawling.                                                                                               |
| `Lampiran_Data_Bersih.csv` | Dataset yang telah melalui proses pembersihan data.                                                                          |
| `korpus_pendidikan/`       | Folder penyimpanan data korpus.                                                                                              |
| `foto/`                    | Folder berisi foto anggota tim (Dea, Dinda, Tasya).                                                                          |

---

## 5. Tim Pengembang

Proyek ini dikembangkan oleh **Kelompok 10**, yang terdiri dari:

* Adinda Muarriva (2308107010001)
* Dea Zasqia Pasaribu Malau (2308107010004)
* Tasya Zahrani (2308107010006)

---
