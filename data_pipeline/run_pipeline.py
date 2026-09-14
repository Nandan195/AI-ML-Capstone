import os
import logging
import pandas as pd
from scrape_pipeline import run_scraping_pipeline
from database import setup_database, load_data
from queries import execute_queries, verify_pandas_merge

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

def main():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        
    logging.info("=== STEP 1: Scraping Data ===")
    df = run_scraping_pipeline(target_books=60, min_categories=3)
    
    row_count = len(df)
    cat_count = df['category'].nunique()
    logging.info(f"VERIFICATION: Total rows = {row_count} (Expected >= 60)")
    logging.info(f"VERIFICATION: Total categories = {cat_count} (Expected >= 3)")
    
    assert row_count >= 60, "Failed to scrape at least 60 books"
    assert cat_count >= 3, "Failed to scrape at least 3 categories"
    
    # Check types
    assert pd.api.types.is_float_dtype(df['price_gbp']), "price_gbp must be float"
    assert pd.api.types.is_integer_dtype(df['rating']), "rating must be integer"
    assert pd.api.types.is_bool_dtype(df['in_stock']), "in_stock must be boolean"
    
    logging.info("=== STEP 2: Database Load ===")
    setup_database()
    load_data(df)
    
    logging.info("=== STEP 3: Execute SQL Queries ===")
    results = execute_queries()
    
    for query_name, result_df in results.items():
        output_path = os.path.join(OUTPUTS_DIR, f"{query_name}.csv")
        result_df.to_csv(output_path, index=False)
        logging.info(f"Saved {query_name} to {output_path}")
        
    logging.info("=== STEP 4: Verify SQL JOIN vs Pandas Merge ===")
    sql_join_result = results['q5_join_books_categories']
    pandas_merge_result = verify_pandas_merge(df)
    
    # We compare the shapes and the first few rows, or we can use pandas testing framework
    try:
        pd.testing.assert_frame_equal(sql_join_result, pandas_merge_result, check_dtype=False)
        logging.info("VERIFICATION SUCCESS: SQL JOIN result perfectly matches Pandas merge result.")
    except AssertionError as e:
        logging.error("VERIFICATION FAILED: SQL JOIN and Pandas merge results differ.")
        logging.error(e)
        
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
