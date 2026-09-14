import sqlite3
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DB_DIR = os.path.join(os.path.dirname(__file__), 'database')
DB_PATH = os.path.join(DB_DIR, 'zepto_books.db')

def execute_queries():
    """Executes the required SQL queries and returns the results."""
    
    if not os.path.exists(DB_PATH):
        logging.error(f"Database not found at {DB_PATH}. Run pipeline first.")
        return None
        
    conn = sqlite3.connect(DB_PATH)
    results = {}
    
    # 1. SELECT, ORDER BY, LIMIT
    # Top 5 most expensive books in GBP
    q1 = """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5;
    """
    results['q1_expensive_books'] = pd.read_sql(q1, conn)
    
    # 2. DISTINCT, WHERE
    # Distinct categories (by id) that have books rated 4 or 5
    q2 = """
        SELECT DISTINCT category_id
        FROM books
        WHERE rating >= 4;
    """
    results['q2_distinct_categories_high_rating'] = pd.read_sql(q2, conn)
    
    # 3. BETWEEN
    # Books with rating between 3 and 5 (inclusive)
    q3 = """
        SELECT title, rating
        FROM books
        WHERE rating BETWEEN 3 AND 5;
    """
    results['q3_books_rating_3_to_5'] = pd.read_sql(q3, conn)
    
    # 4. IN
    # Books in a specific set of ratings
    q4 = """
        SELECT title, rating
        FROM books
        WHERE rating IN (1, 2);
    """
    results['q4_books_rating_1_or_2'] = pd.read_sql(q4, conn)
    
    # 5. JOIN
    # Join books and categories
    q5 = """
        SELECT b.title, b.price_gbp, c.name as category_name
        FROM books b
        JOIN categories c ON b.category_id = c.id
        ORDER BY b.price_gbp DESC;
    """
    results['q5_join_books_categories'] = pd.read_sql(q5, conn)
    
    conn.close()
    return results

def verify_pandas_merge(df: pd.DataFrame):
    """
    Reproduces the JOIN result using pd.merge() on in-memory DataFrames.
    df: The initial dataframe before database insertion.
    """
    # Create an explicit categories dataframe matching the DB table logic
    categories_df = pd.DataFrame(df['category'].unique(), columns=['name'])
    categories_df['id'] = range(1, len(categories_df) + 1)
    
    # Create books dataframe mapping category names to IDs
    books_df = df.copy()
    books_df = books_df.merge(categories_df, left_on='category', right_on='name', how='left')
    books_df = books_df.rename(columns={'id': 'category_id'})
    
    # Now reproduce Query 5 using pandas merge
    # We want: title, price_gbp, category_name (which is 'name' in categories_df)
    merged = pd.merge(
        books_df[['title', 'price_gbp', 'category_id']], 
        categories_df[['id', 'name']], 
        left_on='category_id', 
        right_on='id', 
        how='inner'
    )
    
    # Select columns and sort identically to Q5
    merged = merged[['title', 'price_gbp', 'name']]
    merged = merged.rename(columns={'name': 'category_name'})
    merged = merged.sort_values(by='price_gbp', ascending=False).reset_index(drop=True)
    
    return merged

if __name__ == "__main__":
    res = execute_queries()
    if res:
        for k, v in res.items():
            print(f"--- {k} ---")
            print(v.head())
