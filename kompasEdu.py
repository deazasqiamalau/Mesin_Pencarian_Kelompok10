import requests
from bs4 import BeautifulSoup
import csv
import time
import random

BASE_URL = "https://www.kompas.com/edu?page="
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0 Safari/537.36"
}

# -------------------------------------------------------------------
def safe_get(url):
    url = url.replace("http://", "https://")

    for _ in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r
        except:
            time.sleep(1)
    return None

# -------------------------------------------------------------------
def get_links_from_page(page):
    print(f"\nScraping page: {page}")
    url = BASE_URL + str(page)
    r = safe_get(url)

    if not r:
        print("   ❌ Gagal load halaman")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    all_a = soup.find_all("a", href=True)

    links = []
    for a in all_a:
        href = a["href"]

        if "kompas.com/edu/read/" in href or "edukasi.kompas.com/read/" in href:

            # Normalisasi semua bentuk URL
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/"):
                href = "https://www.kompas.com" + href
            if href.startswith("http://"):
                href = href.replace("http://", "https://")

            links.append(href)

    links = list(set(links))
    print(f"   ➤ Link ditemukan: {len(links)}")
    return links

# -------------------------------------------------------------------
def extract_article(url):
    print(f"   ➤ Ambil artikel: {url}")

    r = safe_get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Publish date
    date_tag = soup.find("div", class_="read__time")
    publish_date = date_tag.get_text(strip=True) if date_tag else "Unknown"

    # Content
    paragraphs = soup.find_all("p")
    content = " ".join([p.get_text(strip=True) for p in paragraphs])

    return title, publish_date, content

# -------------------------------------------------------------------
def save_csv(data, filename="kompasEdu.csv"):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "title", "source", "url", "publish_date", "content"])
        writer.writerows(data)
    print(f"\n📁 CSV tersimpan: {filename}")

# -------------------------------------------------------------------
all_articles = []
doc_id = 1

for page in range(1, 6):
    urls = get_links_from_page(page)

    for url in urls:
        time.sleep(random.uniform(1.2, 2.0))

        extracted = extract_article(url)
        if not extracted:
            continue

        title, publish_date, content = extracted

        all_articles.append([
            doc_id,
            title,
            "kompas",
            url,
            publish_date,
            content
        ])
        doc_id += 1

print(f"\nSelesai! Total artikel: {len(all_articles)}")
save_csv(all_articles)
