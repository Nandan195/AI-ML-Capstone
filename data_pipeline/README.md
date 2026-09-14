# Data Pipeline Module

This module fulfills Module 1 of the capstone project.

## Architecture & Setup
The pipeline uses `requests` and `BeautifulSoup` to scrape book data from `books.toscrape.com`. It processes the data using `pandas`, and stores it in a normalized SQLite database.

**To run the pipeline:**
1. Ensure the virtual environment is active.
2. Run `python run_pipeline.py`

## Cleaning Decisions
- **`price_gbp`**: Raw strings like `Â£51.77` are stripped of non-numeric characters and cast to `float`.
- **`price_inr`**: Calculated as `price_gbp * 105.50` and rounded to 2 decimal places.
- **`rating`**: Extracted from CSS classes (`star-rating One`) and mapped to integer (1-5). Defaults to `0` if unknown.
- **`in_stock`**: Parsed as boolean based on whether 'in stock' appears in the availability string.
- Defensible handling: If a book parsing fails (e.g., missing price tag), the row is skipped, and a warning is logged.

## Schema
Normalized into two tables to reduce redundancy:
1. **`categories`**
   - `id`: INTEGER PRIMARY KEY
   - `name`: TEXT UNIQUE (e.g., 'Travel')
2. **`books`**
   - `id`: INTEGER PRIMARY KEY
   - `title`: TEXT
   - `price_gbp`: REAL
   - `price_inr`: REAL
   - `rating`: INTEGER
   - `in_stock`: BOOLEAN
   - `category_id`: INTEGER (FOREIGN KEY)

## SQL Queries
5 queries are run spanning SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, and JOIN.
Outputs are saved in the `outputs/` folder. Query 5 (a JOIN) is automatically validated against a `pd.merge()` equivalent to guarantee data consistency.
