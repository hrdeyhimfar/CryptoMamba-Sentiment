import requests
import schedule
import time
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
import unicodedata
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
import cloudscraper
import re
from dateutil.parser import parse as parse_date
from urllib.parse import urlparse
from cachetools import TTLCache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# تنظیمات اولیه
COINGECKO_URL = "https://api.coingecko.com/api/v3/news"
MAX_ARTICLES_TO_SCRAPE = 10  # افزایش برای جمع‌آوری مقالات بیشتر
SCRAPE_TIMEOUT = 20  # افزایش تایم‌اوت
DELAY_BETWEEN_REQUESTS = 0.5

# تنظیمات اتصال به دیتابیس
DB_CONFIG = {
    "host": "127.0.0.1",  # هاست دیتابیس
    "user": "root",  # نام کاربری MariaDB
    "password": "Hh12345678",  # رمز عبور MariaDB
    "database": "crypto_news_db",
    "charset": "utf8mb4",
    "collation": "utf8mb4_unicode_ci",
    "pool_name": "news_pool",
    "pool_size": 5
}

# کش برای پاسخ API
cache = TTLCache(maxsize=100, ttl=3600)

# هدرهای متنوع برای اسکریپینگ
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

# بارگذاری مدل FinBERT
try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    print("FinBERT model loaded successfully.")
except Exception as e:
    print(f"Error loading FinBERT model: {e}")
    tokenizer = None
    model = None

# تنظیمات برای تلاش مجدد درخواست‌های API
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504, 422])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# اتصال به دیتابیس
def connect_to_db():
    try:
        pool = MySQLConnectionPool(**DB_CONFIG)
        connection = pool.get_connection()
        if connection.is_connected():
            print("Connected to MariaDB database via connection pool.")
            return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

# پاک‌سازی متن
def clean_text(text):
    if not text:
        return None
    try:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text[:1000]
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return None

# استخراج source از URL
def extract_source_from_url(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        domain = re.sub(r'^www\.', '', domain)
        domain = re.sub(r'\.\w+$', '', domain)
        return domain.capitalize() if domain else "Unknown"
    except Exception as e:
        print(f"Error extracting source from URL {url}: {e}")
        return "Unknown"

# اسکریپینگ غیرهمزمان بهبودیافته
async def scrape_article_content_async(url, session):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        scraper = cloudscraper.create_scraper()
        async with session.get(url, headers=headers, timeout=SCRAPE_TIMEOUT) as response:
            response.raise_for_status()
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")
            content = ""
            for tag in ['p', 'article', 'div[class*="content"]', 'div[class*="article"]', 'div[class*="post"]', 'section']:
                elements = soup.select(tag)
                content += " ".join(el.get_text(strip=True) for el in elements if el.get_text(strip=True))
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            return clean_text(content) if content else "Content unavailable"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Scraping error for {url}: {e}")
        return clean_text(f"Scraping error: {str(e)}")
    except Exception as e:
        print(f"Processing error for {url}: {e}")
        return clean_text(f"Processing error: {str(e)}")

# تحلیل احساسات با FinBERT
def analyze_sentiment(texts):
    if not texts or not tokenizer or not model:
        return [("neutral", 0.0)] * len(texts)
    
    try:
        valid_texts = [text for text in texts if text and text.strip() and text != "Content unavailable" and text != "No description"]
        invalid_count = len(texts) - len(valid_texts)
        print(f"Valid texts: {len(valid_texts)}, Invalid texts: {invalid_count}")
        
        if not valid_texts:
            return [("neutral", 0.0)] * len(texts)
        
        inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        labels = ["positive", "negative", "neutral"]
        results = []
        valid_idx = 0
        for i in range(len(texts)):
            if texts[i] and texts[i].strip() and texts[i] != "Content unavailable" and texts[i] != "No description":
                max_idx = torch.argmax(probs[valid_idx]).item()
                score = probs[valid_idx][max_idx].item()
                label = labels[max_idx]
                results.append((label, score))
                valid_idx += 1
            else:
                results.append(("neutral", 0.0))
        return results
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return [("neutral", 0.0)] * len(texts)

# بررسی گروهی لینک‌های موجود
def links_exist(connection, links):
    try:
        cursor = connection.cursor()
        query = "SELECT link FROM news WHERE link IN (%s)" % ','.join(['%s'] * len(links))
        cursor.execute(query, links)
        existing_links = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return existing_links
    except Error as e:
        print(f"Error checking links: {e}")
        return set()

# ذخیره گروهی مقالات
def save_articles_to_db(connection, articles):
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO news (title, source, pub_date, link, description, full_content, sentiment_label, sentiment_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(query, [
            (
                clean_text(article["title"]),
                clean_text(article["source"]),
                article["pub_date"],
                clean_text(article["link"]),
                clean_text(article["description"]),
                clean_text(article["full_content"]),
                article["sentiment_label"],
                article["sentiment_score"]
            ) for article in articles
        ])
        connection.commit()
        cursor.close()
        return True
    except Error as e:
        print(f"Error saving articles: {e}")
        connection.rollback()
        return False

# دریافت و ذخیره اخبار
async def fetch_and_store_news():
    connection = connect_to_db()
    if not connection:
        print("Failed to connect to database. Stopping.")
        return
    
    session = create_session()
    new_articles = []
    duplicate_count = 0
    
    cache_key = "coingecko_news"
    if cache_key in cache:
        print("Using cached CoinGecko data.")
        news_data = cache[cache_key]
    else:
        try:
            print(f"Fetching news from {COINGECKO_URL}...")
            response = session.get(COINGECKO_URL, timeout=30)
            if response.status_code != 200:
                print(f"HTTP error: Status code {response.status_code}")
                print(f"Response content: {response.text}")
                news_data = {"data": []}
            else:
                news_data = response.json()
                cache[cache_key] = news_data
                print(f"Raw API response (first article): {news_data.get('data', [])[:1]}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            news_data = {"data": []}
    
    print(f"Fetched news at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total results from API: {len(news_data.get('data', []))}")
    
    articles = news_data.get("data", [])[:MAX_ARTICLES_TO_SCRAPE]
    links = [article.get("url", "") for article in articles if article.get("url")]
    
    if not links:
        print("No valid links found.")
        if connection.is_connected():
            connection.close()
        return
    
    existing_links = links_exist(connection, links)
    
    new_articles = []
    texts_to_analyze = []
    async with aiohttp.ClientSession() as async_session:
        for article in articles:
            link = article.get("url", "")
            if not link or link in existing_links:
                print(f"Article with link {link} already exists or invalid.")
                duplicate_count += 1
                continue
            source = (article.get("source") or 
                     article.get("news_site") or 
                     article.get("publisher") or 
                     extract_source_from_url(link))
            pub_date = article.get("date", None)
            if pub_date:
                try:
                    pub_date = parse_date(pub_date)
                except (ValueError, TypeError) as e:
                    print(f"Error parsing pub_date for {link}: {e}")
                    pub_date = None
            full_content = await scrape_article_content_async(link, async_session)
            description = article.get("description", "No description") or "No description"
            text_to_analyze = clean_text(full_content) or clean_text(description) or "Content unavailable"
            new_articles.append({
                "title": article.get("title", "No title"),
                "source": source,
                "pub_date": pub_date if pub_date else datetime.now(),
                "link": link,
                "description": description,
                "full_content": full_content,
                "sentiment_label": "neutral",
                "sentiment_score": 0.0
            })
            texts_to_analyze.append(text_to_analyze)
    
    if new_articles and texts_to_analyze:
        print(f"Analyzing sentiment for {len(texts_to_analyze)} articles...")
        sentiment_results = analyze_sentiment(texts_to_analyze)
        for article, (label, score) in zip(new_articles, sentiment_results):
            article["sentiment_label"] = label
            article["sentiment_score"] = score
    
    print(f"Duplicate articles skipped: {duplicate_count}")
    if new_articles:
        print(f"Attempting to save {len(new_articles)} new articles...")
        if save_articles_to_db(connection, new_articles):
            print(f"{len(new_articles)} new articles saved to database.")
            print("Saved articles:")
            for article in new_articles[:5]:
                print(f"- Title: {article['title']}")
                print(f"  Source: {article['source']}")
                print(f"  Pub Date: {article['pub_date']}")
                print(f"  Sentiment: {article['sentiment_label']} (Score: {article['sentiment_score']:.2f})")
                print(f"  Description: {article['description'][:100]}...")
                print(f"  Full Content (first 100 chars): {article['full_content'][:100]}...")
        else:
            print("Failed to save articles.")
    else:
        print("No new articles to save.")
    
    if connection.is_connected():
        connection.close()
        print("Database connection closed.")

# تنظیم برنامه‌ریزی
def main():
    print("Starting cryptocurrency news fetcher with improved scraping and FinBERT analysis (every 10 minutes)...")
    schedule.every(10).minutes.do(lambda: asyncio.run(fetch_and_store_news()))
    asyncio.run(fetch_and_store_news())  # اجرای اولیه
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
