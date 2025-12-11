import requests
import json
import os
import csv
from datetime import datetime
import time
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PendidikanCorpusCrawler:
    """
    Crawler khusus untuk korpus pendidikan Indonesia
    dengan error handling yang lebih baik
    """
    
    def __init__(self, output_dir="korpus_pendidikan"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup session dengan retry mechanism
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def crawl_wikipedia(self, categories, max_per_category=60):
        """Wikipedia Indonesia API - Fokus Pendidikan"""
        all_documents = []
        base_url = "https://id.wikipedia.org/w/api.php"
        
        for category in categories:
            print(f"\n📚 Crawling kategori: {category}")
            documents = []
            
            try:
                # Step 1: Ambil daftar artikel dalam kategori
                params = {
                    'action': 'query',
                    'list': 'categorymembers',
                    'cmtitle': f'Kategori:{category}',
                    'cmlimit': 500,
                    'cmtype': 'page',
                    'format': 'json',
                    'formatversion': '2'
                }
                
                print(f"  🔍 Mengambil daftar artikel...")
                response = self.session.get(base_url, params=params, timeout=15, verify=False)
                response.raise_for_status()
                
                data = response.json()
                
                if 'query' not in data:
                    print(f"  ⚠️ Kategori tidak ditemukan atau kosong")
                    continue
                
                pages = data.get('query', {}).get('categorymembers', [])
                print(f"  📄 Ditemukan {len(pages)} artikel dalam kategori")
                
                # Step 2: Ambil konten setiap artikel
                for i, page in enumerate(pages[:max_per_category], 1):
                    try:
                        page_title = page.get('title', '')
                        
                        # Skip jika bukan artikel (file, template, dll)
                        if any(x in page_title for x in ['Berkas:', 'File:', 'Template:', 'Kategori:', 'Wikipedia:']):
                            continue
                        
                        # Ambil konten artikel
                        content_params = {
                            'action': 'query',
                            'titles': page_title,
                            'prop': 'extracts|info',
                            'explaintext': True,
                            'exsectionformat': 'plain',
                            'inprop': 'url',
                            'format': 'json',
                            'formatversion': '2'
                        }
                        
                        content_response = self.session.get(base_url, params=content_params, timeout=15, verify=False)
                        content_response.raise_for_status()
                        content_data = content_response.json()
                        
                        pages_data = content_data.get('query', {}).get('pages', [])
                        
                        if not pages_data:
                            continue
                        
                        page_data = pages_data[0]
                        content = page_data.get('extract', '')
                        
                        # Validasi konten
                        if content and len(content) > 500:
                            # Split menjadi paragraf
                            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
                            
                            if len(paragraphs) >= 3:
                                doc = {
                                    'id': f"wiki_{page.get('pageid', i)}",
                                    'source': 'Wikipedia Indonesia',
                                    'category': category,
                                    'url': page_data.get('canonicalurl', ''),
                                    'title': page_data.get('title', ''),
                                    'content': content,
                                    'paragraph_count': len(paragraphs),
                                    'word_count': len(content.split()),
                                    'crawled_date': datetime.now().isoformat()
                                }
                                
                                documents.append(doc)
                                print(f"  ✓ [{len(documents):3d}] {doc['title'][:55]}...")
                        
                        # Rate limiting
                        if i % 10 == 0:
                            time.sleep(1)
                        else:
                            time.sleep(0.3)
                    
                    except Exception as e:
                        print(f"  ⚠️ Skip artikel: {str(e)[:50]}")
                        continue
                
                all_documents.extend(documents)
                print(f"  📊 Total dari kategori {category}: {len(documents)} dokumen")
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error koneksi pada {category}: {str(e)[:80]}")
            except Exception as e:
                print(f"  ❌ Error pada {category}: {str(e)[:80]}")
        
        return all_documents
    
    def crawl_kompas_pendidikan(self, max_docs=100):
        """
        Crawl artikel pendidikan dari Kompas.com
        """
        documents = []
        base_url = "https://www.kompas.com"
        search_url = f"{base_url}/search/pendidikan"
        
        print(f"\n📰 Crawling Kompas.com (Pendidikan)...")
        
        try:
            for page in range(1, (max_docs // 15) + 2):
                try:
                    url = f"{search_url}?page={page}"
                    response = self.session.get(url, timeout=15, verify=False)
                    response.raise_for_status()
                    
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Cari link artikel
                    articles = soup.find_all('div', class_='article__list')
                    if not articles:
                        articles = soup.find_all('article')
                    
                    for article in articles:
                        if len(documents) >= max_docs:
                            break
                        
                        link_tag = article.find('a', href=True)
                        if link_tag and link_tag['href'].startswith('http'):
                            article_url = link_tag['href']
                            
                            # Skip jika bukan artikel berita
                            if 'kompas.com' not in article_url:
                                continue
                            
                            doc = self.extract_kompas_article(article_url)
                            if doc and len(doc.get('content', '')) > 500:
                                documents.append(doc)
                                print(f"  ✓ [{len(documents):3d}] {doc['title'][:55]}...")
                    
                    time.sleep(2)  # Rate limiting
                    
                    if len(documents) >= max_docs:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Skip halaman {page}: {str(e)[:50]}")
                    continue
            
            print(f"  📊 Total dari Kompas: {len(documents)} dokumen")
            
        except Exception as e:
            print(f"  ❌ Error Kompas: {str(e)[:80]}")
        
        return documents
    
    def extract_kompas_article(self, url):
        """Ekstrak konten artikel dari Kompas"""
        try:
            response = self.session.get(url, timeout=10, verify=False)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Judul
            title_tag = soup.find('h1', class_='read__title')
            if not title_tag:
                title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else ''
            
            # Tanggal
            date_tag = soup.find('div', class_='read__time')
            publish_date = date_tag.get_text().strip() if date_tag else datetime.now().strftime('%Y-%m-%d')
            
            # Konten
            paragraphs = []
            content_div = soup.find('div', class_='read__content')
            if content_div:
                for p in content_div.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 50:
                        paragraphs.append(text)
            
            if not paragraphs or not title:
                return None
            
            content = '\n\n'.join(paragraphs)
            
            return {
                'source': 'Kompas',
                'title': title,
                'content': content,
                'url': url,
                'publish_date': publish_date,
                'word_count': len(content.split()),
                'crawled_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
    
    def crawl_detik_pendidikan(self, max_docs=100):
        """
        Crawl artikel pendidikan dari Detik.com
        """
        documents = []
        search_url = "https://www.detik.com/search/searchall"
        
        print(f"\n📰 Crawling Detik.com (Pendidikan)...")
        
        try:
            for page in range(1, (max_docs // 10) + 2):
                try:
                    params = {
                        'query': 'pendidikan',
                        'sortby': 'time',
                        'page': page
                    }
                    
                    response = self.session.get(search_url, params=params, timeout=15, verify=False)
                    response.raise_for_status()
                    
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Cari link artikel
                    articles = soup.find_all('article')
                    
                    for article in articles:
                        if len(documents) >= max_docs:
                            break
                        
                        link_tag = article.find('a', href=True)
                        if link_tag:
                            article_url = link_tag['href']
                            
                            if 'detik.com' not in article_url:
                                continue
                            
                            doc = self.extract_detik_article(article_url)
                            if doc and len(doc.get('content', '')) > 500:
                                documents.append(doc)
                                print(f"  ✓ [{len(documents):3d}] {doc['title'][:55]}...")
                    
                    time.sleep(2)
                    
                    if len(documents) >= max_docs:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Skip halaman {page}: {str(e)[:50]}")
                    continue
            
            print(f"  📊 Total dari Detik: {len(documents)} dokumen")
            
        except Exception as e:
            print(f"  ❌ Error Detik: {str(e)[:80]}")
        
        return documents
    
    def extract_detik_article(self, url):
        """Ekstrak konten artikel dari Detik"""
        try:
            response = self.session.get(url, timeout=10, verify=False)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Judul
            title_tag = soup.find('h1', class_='detail__title')
            if not title_tag:
                title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else ''
            
            # Tanggal
            date_tag = soup.find('div', class_='detail__date')
            publish_date = date_tag.get_text().strip() if date_tag else datetime.now().strftime('%Y-%m-%d')
            
            # Konten
            paragraphs = []
            content_div = soup.find('div', class_='detail__body-text')
            if content_div:
                for p in content_div.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 50:
                        paragraphs.append(text)
            
            if not paragraphs or not title:
                return None
            
            content = '\n\n'.join(paragraphs)
            
            return {
                'source': 'Detik',
                'title': title,
                'content': content,
                'url': url,
                'publish_date': publish_date,
                'word_count': len(content.split()),
                'crawled_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
    
    def crawl_republika_pendidikan(self, max_docs=100):
        """
        Crawl artikel pendidikan dari Republika.co.id
        """
        documents = []
        search_url = "https://www.republika.co.id/search"
        
        print(f"\n📰 Crawling Republika.co.id (Pendidikan)...")
        
        try:
            for page in range(1, (max_docs // 10) + 2):
                try:
                    params = {
                        'q': 'pendidikan',
                        'page': page
                    }
                    
                    response = self.session.get(search_url, params=params, timeout=15, verify=False)
                    response.raise_for_status()
                    
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Cari link artikel
                    articles = soup.find_all('article')
                    if not articles:
                        articles = soup.find_all('div', class_='artikel')
                    
                    for article in articles:
                        if len(documents) >= max_docs:
                            break
                        
                        link_tag = article.find('a', href=True)
                        if link_tag:
                            article_url = link_tag['href']
                            
                            if not article_url.startswith('http'):
                                article_url = f"https://www.republika.co.id{article_url}"
                            
                            if 'republika.co.id' not in article_url:
                                continue
                            
                            doc = self.extract_republika_article(article_url)
                            if doc and len(doc.get('content', '')) > 500:
                                documents.append(doc)
                                print(f"  ✓ [{len(documents):3d}] {doc['title'][:55]}...")
                    
                    time.sleep(2)
                    
                    if len(documents) >= max_docs:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Skip halaman {page}: {str(e)[:50]}")
                    continue
            
            print(f"  📊 Total dari Republika: {len(documents)} dokumen")
            
        except Exception as e:
            print(f"  ❌ Error Republika: {str(e)[:80]}")
        
        return documents
    
    def extract_republika_article(self, url):
        """Ekstrak konten artikel dari Republika"""
        try:
            response = self.session.get(url, timeout=10, verify=False)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Judul
            title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else ''
            
            # Tanggal
            date_tag = soup.find('time')
            if date_tag:
                publish_date = date_tag.get('datetime', datetime.now().strftime('%Y-%m-%d'))
            else:
                publish_date = datetime.now().strftime('%Y-%m-%d')
            
            # Konten
            paragraphs = []
            content_div = soup.find('div', class_='article-content')
            if not content_div:
                content_div = soup.find('div', {'id': 'content'})
            
            if content_div:
                for p in content_div.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 50:
                        paragraphs.append(text)
            
            if not paragraphs or not title:
                return None
            
            content = '\n\n'.join(paragraphs)
            
            return {
                'source': 'Republika',
                'title': title,
                'content': content,
                'url': url,
                'publish_date': publish_date,
                'word_count': len(content.split()),
                'crawled_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None

        """
        Reddit dengan old.reddit.com yang lebih stabil
        """
        documents = []
        
        print(f"\n💬 Crawling Reddit: r/{subreddit} (topik: {search_query})")
        
        try:
            # Gunakan old.reddit untuk lebih stabil
            url = f"https://old.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': search_query,
                'restrict_sr': 'on',
                'sort': 'relevance',
                'limit': 100
            }
            
            response = self.session.get(url, params=params, timeout=15, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            for post in data.get('data', {}).get('children', [])[:max_posts]:
                post_data = post.get('data', {})
                
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                content = f"{title}\n\n{selftext}"
                
                # Filter: harus ada konten yang cukup panjang
                if len(content) > 300 and not post_data.get('over_18', False):
                    doc = {
                        'id': f"reddit_{post_data.get('id')}",
                        'source': 'Reddit Indonesia',
                        'category': f'r/{subreddit}',
                        'title': title,
                        'content': content,
                        'author': post_data.get('author', 'unknown'),
                        'score': post_data.get('score', 0),
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'word_count': len(content.split()),
                        'crawled_date': datetime.now().isoformat()
                    }
                    
                    documents.append(doc)
                    print(f"  ✓ [{len(documents):3d}] {doc['title'][:55]}...")
            
            print(f"  📊 Total dari Reddit: {len(documents)} dokumen")
            
        except Exception as e:
            print(f"  ⚠️ Reddit error (skip): {str(e)[:80]}")
        
        return documents
    
    def save_to_csv(self, documents, filename="korpus_pendidikan.csv"):
        """Simpan dokumen ke CSV dengan format standar"""
        if not documents:
            print("❌ Tidak ada dokumen untuk disimpan")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Kolom CSV sesuai requirement
        fieldnames = ['doc_id', 'title', 'source', 'url', 'publish_date', 'content']
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Counter per sumber untuk ID
            source_counters = {}
            
            for doc in documents:
                # Generate doc_id dengan format SUMBER_XXX
                source_name = doc.get('source', 'UNKNOWN').upper().replace(' ', '_')
                
                # Ekstrak nama sumber singkat
                if 'WIKIPEDIA' in source_name:
                    source_prefix = 'WIKIPEDIA'
                elif 'REDDIT' in source_name:
                    source_prefix = 'REDDIT'
                elif 'KOMPAS' in source_name:
                    source_prefix = 'KOMPAS'
                elif 'DETIK' in source_name:
                    source_prefix = 'DETIK'
                elif 'REPUBLIKA' in source_name:
                    source_prefix = 'REPUBLIKA'
                else:
                    source_prefix = source_name.split('_')[0][:10]
                
                # Increment counter untuk source ini
                if source_prefix not in source_counters:
                    source_counters[source_prefix] = 0
                source_counters[source_prefix] += 1
                
                # Format doc_id: WIKIPEDIA_001, REDDIT_001, dll
                doc_id = f"{source_prefix}_{source_counters[source_prefix]:03d}"
                
                # Bersihkan content dari newline untuk CSV
                content = doc.get('content', '').replace('\n', ' ').replace('\r', ' ')
                
                # Ambil publish_date atau gunakan crawled_date
                publish_date = doc.get('publish_date') or doc.get('published_at') or doc.get('crawled_date', '')
                if publish_date:
                    # Format tanggal jika perlu
                    try:
                        if 'T' in publish_date:
                            publish_date = publish_date.split('T')[0]
                    except:
                        pass
                
                row = {
                    'doc_id': doc_id,
                    'title': doc.get('title', ''),
                    'source': doc.get('source', ''),
                    'url': doc.get('url', ''),
                    'publish_date': publish_date,
                    'content': content
                }
                
                writer.writerow(row)
        
        print(f"\n✅ CSV BERHASIL DISIMPAN")
        print(f"📁 File: {filepath}")
        print(f"📊 Total Dokumen: {len(documents)}")
        print(f"\n📋 Format CSV:")
        print(f"  • doc_id: ID unik (format: WIKIPEDIA_001)")
        print(f"  • title: Judul artikel")
        print(f"  • source: Sumber (Wikipedia/Reddit/dll)")
        print(f"  • url: URL artikel asli")
        print(f"  • publish_date: Tanggal publikasi")
        print(f"  • content: Isi lengkap artikel")
        
        return filepath
    
    def save_to_json(self, documents, filename="korpus_pendidikan.json"):
        """Simpan dokumen ke JSON dengan metadata"""
        if not documents:
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        corpus = {
            'metadata': {
                'total_documents': len(documents),
                'created_date': datetime.now().isoformat(),
                'topic': 'Pendidikan Indonesia',
                'sources': list(set(doc.get('source', 'Unknown') for doc in documents)),
                'total_words': sum(doc.get('word_count', 0) for doc in documents),
                'average_words_per_doc': sum(doc.get('word_count', 0) for doc in documents) / len(documents) if documents else 0
            },
            'documents': documents
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ JSON BERHASIL DISIMPAN")
        print(f"📁 File: {filepath}")
        
        return filepath
    
    def show_statistics(self, documents):
        """Tampilkan statistik korpus"""
        if not documents:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 STATISTIK KORPUS PENDIDIKAN")
        print(f"{'='*80}")
        
        # Hitung per sumber
        source_count = {}
        category_count = {}
        total_words = 0
        
        for doc in documents:
            source = doc.get('source', 'Unknown')
            category = doc.get('category', 'Unknown')
            source_count[source] = source_count.get(source, 0) + 1
            category_count[category] = category_count.get(category, 0) + 1
            total_words += doc.get('word_count', 0)
        
        print(f"\n📚 Total Dokumen: {len(documents)}")
        print(f"📝 Total Kata: {total_words:,}")
        print(f"📈 Rata-rata Kata/Dokumen: {total_words/len(documents):.0f}")
        
        print(f"\n🔗 Distribusi per Sumber:")
        for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {source}: {count} dokumen ({count/len(documents)*100:.1f}%)")
        
        print(f"\n📑 Distribusi per Kategori (Top 5):")
        for category, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {category}: {count} dokumen")

# ========== MAIN PROGRAM ==========
if __name__ == "__main__":
    print("="*80)
    print("🎓 CRAWLER KORPUS PENDIDIKAN INDONESIA")
    print("="*80)
    print("\n📋 Sumber: Kompas, Detik, Republika, Wikipedia")
    print("📂 Output: CSV dengan format standar")
    print("🎯 Target: 300+ dokumen tentang pendidikan\n")
    
    # Install BeautifulSoup jika belum ada
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ BeautifulSoup4 belum terinstall!")
        print("📦 Install dengan: pip install beautifulsoup4")
        print("   Kemudian jalankan ulang program ini\n")
        exit(1)
    
    crawler = PendidikanCorpusCrawler()
    all_documents = []
    
    # Crawl dari media online Indonesia
    print("\n🚀 CRAWLING DARI MEDIA ONLINE INDONESIA...\n")
    
    # 1. Kompas.com
    try:
        docs_kompas = crawler.crawl_kompas_pendidikan(max_docs=100)
        all_documents.extend(docs_kompas)
    except Exception as e:
        print(f"⚠️ Kompas skip: {str(e)[:50]}")
    
    # 2. Detik.com  
    try:
        docs_detik = crawler.crawl_detik_pendidikan(max_docs=100)
        all_documents.extend(docs_detik)
    except Exception as e:
        print(f"⚠️ Detik skip: {str(e)[:50]}")
    
    # 3. Republika.co.id
    try:
        docs_republika = crawler.crawl_republika_pendidikan(max_docs=100)
        all_documents.extend(docs_republika)
    except Exception as e:
        print(f"⚠️ Republika skip: {str(e)[:50]}")
    
    # Wikipedia - Kategori Pendidikan (sebagai pelengkap)
    wiki_categories = [
        'Pendidikan_di_Indonesia',
        'Universitas_di_Indonesia', 
        'Sekolah_di_Indonesia',
    ]
    
    docs_wiki = crawler.crawl_wikipedia(wiki_categories, max_per_category=50)
    all_documents.extend(docs_wiki)
    
    # Tampilkan statistik
    if all_documents:
        crawler.show_statistics(all_documents)
        
        # Simpan ke CSV dan JSON
        print(f"\n{'='*80}")
        print("💾 MENYIMPAN FILE...")
        print(f"{'='*80}")
        
        csv_file = crawler.save_to_csv(all_documents)
        json_file = crawler.save_to_json(all_documents)
        
        print(f"\n{'='*80}")
        print("✅ SELESAI!")
        print(f"{'='*80}")
        print(f"\n📂 File tersimpan di folder: {crawler.output_dir}")
        print(f"  1. {os.path.basename(csv_file)} (untuk Excel/Pandas)")
        print(f"  2. {os.path.basename(json_file)} (untuk processing)")
        
    else:
        print("\n❌ TIDAK ADA DOKUMEN TERKUMPUL")
        print("\n💡 Troubleshooting:")
        print("  1. Cek koneksi internet")
        print("  2. Coba jalankan ulang (Wikipedia kadang timeout)")
        print("  3. Pastikan tidak ada firewall yang memblokir")