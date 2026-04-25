import requests
import json
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coingecko_news_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


base_url = "https://api.coingecko.com/api/v3/news"
all_articles = []  # لیست برای ذخیره تمام مقالات
page = 1
max_pages = 10000  # حداکثر صفحات برای جلوگیری از لوپ بی‌نهایت (می‌توانید تغییر دهید)
batch_articles = []
batch_start_page = 1

PAGES_PER_FILE = 10  # ذخیره هر 10 صفحه در یک فایل JSON


# ذخیره مقالات در فایل JSON
def save_to_json(articles, start_page, end_page):
    filename = f'coingecko_news_pages_{start_page}-{end_page}.json'
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"data": articles}, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(articles)} articles to {filename}")
    except Exception as e:
        logger.error(f"Error saving to {filename}: {e}")


while page <= max_pages:
    url = f"{base_url}?page={page}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            continue
        response.raise_for_status()  # بررسی خطاهای HTTP
        data = response.json()
        articles = data.get('data', [])
        if not articles:
            print(f"No more articles at page {page}. Stopping.")
            break
        all_articles.extend(articles)
        print(f"Fetched page {page} with {len(articles)} articles. Total so far: {len(all_articles)}")

        batch_articles.extend(articles)
        all_articles.extend(articles)

        # ذخیره هر 10 صفحه
        if page % PAGES_PER_FILE == 0 or page == max_pages:
            save_to_json(batch_articles, batch_start_page, page)
            batch_articles = []
            batch_start_page = page + 1

        page += 1
        time.sleep(1)  # تأخیر 1 ثانیه برای جلوگیری از rate limit
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page}: {e}")
        time.sleep(5)  # تأخیر قبل از تلاش مجدد
        continue


print(f"Total articles fetched: {len(all_articles)}")
print("All data saved to 'all_coingecko_news.json'")

# ذخیره مقالات باقی‌مانده (اگر تعداد صفحات مضرب 10 نباشد)
if batch_articles:
    save_to_json(batch_articles, batch_start_page, page - 1)
