import sqlite3
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DB_DIR = os.path.join(os.path.dirname(__file__), 'database')
DB_PATH = os.path.join(DB_DIR, 'zepto_books.db')

def setup_database():
    """Creates the SQLite database and initializes the normalized schema."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables to ensure idempotence
    cursor.execute('DROP TABLE IF EXISTS books')
    cursor.execute('DROP TABLE IF EXISTS categories')
    
    # Create categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    # Create books table with foreign key to categories
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock BOOLEAN NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Database schema initialized at {DB_PATH}")

def load_data(df: pd.DataFrame):
    """Loads the cleaned DataFrame into the normalized SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Insert unique categories and map them to category_id
    categories = df['category'].unique()
    for cat in categories:
        cursor.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (cat,))
    conn.commit()
    
    # 2. Fetch category mapping
    cursor.execute('SELECT id, name FROM categories')
    cat_mapping = {name: cat_id for cat_id, name in cursor.fetchall()}
    
    # 3. Insert books
    books_to_insert = []
    for _, row in df.iterrows():
        cat_id = cat_mapping[row['category']]
        books_to_insert.append((
            row['title'],
            row['price_gbp'],
            row['price_inr'],
            row['rating'],
            row['in_stock'],
            cat_id
        ))
        
    cursor.executemany('''
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', books_to_insert)
    
    conn.commit()
    conn.close()
    logging.info(f"Inserted {len(books_to_insert)} books into the database.")

if __name__ == "__main__":
    setup_database()
