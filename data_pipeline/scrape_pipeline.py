import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_URL = "http://books.toscrape.com"

# Word to number mapping for ratings
RATING_MAP = {
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5
}

def get_categories():
    """Scrapes the list of category URLs."""
    response = requests.get(f"{BASE_URL}/index.html")
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # The categories are in the side panel under a ul with class 'nav nav-list'
    category_list = soup.select_one('.nav-list > li > ul')
    categories = []
    if category_list:
        for a_tag in category_list.find_all('a'):
            cat_name = a_tag.text.strip()
            cat_url = f"{BASE_URL}/{a_tag['href']}"
            categories.append((cat_name, cat_url))
    return categories

def scrape_category_books(cat_name, cat_url, max_books=None):
    """Scrapes books from a specific category."""
    books_data = []
    current_url = cat_url
    
    while current_url:
        logging.info(f"Scraping category '{cat_name}' page: {current_url}")
        try:
            response = requests.get(current_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch {current_url}: {e}")
            break
            
        soup = BeautifulSoup(response.content, 'html.parser')
        book_articles = soup.find_all('article', class_='product_pod')
        
        for article in book_articles:
            if max_books and len(books_data) >= max_books:
                return books_data
                
            try:
                # Title
                title = article.h3.a['title']
                
                # Price (e.g., '£51.77')
                price_text = article.find('p', class_='price_color').text
                
                # Star rating (e.g., 'star-rating Three')
                rating_p = article.find('p', class_='star-rating')
                star_class = [c for c in rating_p['class'] if c != 'star-rating'][0]
                
                # Availability (e.g., 'In stock')
                availability = article.find('p', class_='instock availability').text.strip()
                
                books_data.append({
                    'title': title,
                    'price': price_text,
                    'star_rating': star_class,
                    'availability': availability,
                    'category': cat_name
                })
            except Exception as e:
                logging.warning(f"Failed to parse a book in category {cat_name}: {e}. Skipping book.")
                continue
                
        # Check for next page
        next_btn = soup.find('li', class_='next')
        if next_btn:
            next_url_part = next_btn.a['href']
            # construct absolute url
            current_url = current_url.rsplit('/', 1)[0] + '/' + next_url_part
        else:
            current_url = None
            
        time.sleep(0.1) # Be nice to the server
        
    return books_data

def clean_data(raw_data):
    """Cleans the raw scraped data into required formats."""
    cleaned = []
    for row in raw_data:
        try:
            # 1. Price -> float (remove currency symbol)
            # Example price: '£51.77' or 'Â£51.77'
            price_str = row['price'].replace('£', '').replace('Â', '').strip()
            price_gbp = float(price_str)
            
            # 2. Convert GBP to INR (1 GBP = 105.50 INR)
            price_inr = round(price_gbp * 105.50, 2)
            
            # 3. Rating -> integer (1-5)
            rating = RATING_MAP.get(row['star_rating'], 0)
            
            # 4. in_stock -> boolean
            in_stock = 'in stock' in row['availability'].lower()
            
            cleaned.append({
                'title': row['title'],
                'price_gbp': price_gbp,
                'price_inr': price_inr,
                'rating': rating,
                'in_stock': in_stock,
                'category': row['category']
            })
        except Exception as e:
            logging.error(f"Error cleaning row {row}: {e}. Row skipped.")
            
    return pd.DataFrame(cleaned)

def run_scraping_pipeline(target_books=60, min_categories=3):
    logging.info("Starting scrape pipeline...")
    categories = get_categories()
    
    all_raw_books = []
    categories_scraped = 0
    
    for cat_name, cat_url in categories:
        if len(all_raw_books) >= target_books and categories_scraped >= min_categories:
            break
            
        books = scrape_category_books(cat_name, cat_url)
        all_raw_books.extend(books)
        categories_scraped += 1
        
    logging.info(f"Scraped {len(all_raw_books)} books across {categories_scraped} categories.")
    
    df = clean_data(all_raw_books)
    logging.info(f"Cleaned data into DataFrame with {len(df)} rows.")
    return df

if __name__ == "__main__":
    df = run_scraping_pipeline()
    print(df.head())
    print(df.info())
