# Module 1 — Zepto Data & AI Platform: Data Pipeline

This module implements the Zepto book data pipeline required for the capstone.

## What it does

1. Scrapes the first 5 pagination pages from `books.toscrape.com`.
2. Collects title, price, star rating, availability, and category.
3. Cleans the data into the required types.
4. Converts GBP to INR using the fixed project rate:
   **1 GBP = 105.50 INR**
5. Stores the data in a normalized SQLite database with:
   - `categories`
   - `books`
6. Runs six SQL queries covering:
   - SELECT + WHERE
   - ORDER BY + LIMIT
   - DISTINCT
   - BETWEEN
   - IN
   - JOIN
7. Uses `pandas.read_sql` and reproduces the JOIN using `pandas.merge`.

## Folder structure

```text
data-pipeline/
├── scrap.py
├── clean.py
├── database.py
├── query.py
├── requirements.txt
├── README.md
├── data/
│   ├── book.csv
│   ├── book_clean.csv
│   └── book.db
└── outputs/
    ├── query_outputs.txt
    ├── read_sql_results.csv
    └── merge_results.csv
```

## Setup

From the `data-pipeline` folder:

```powershell
pip install -r requirements.txt
```

## Run

Run the scripts in this order:

```powershell
python scrap.py
python clean.py
python database.py
python query.py
```

The generated files are written into `data/` and `outputs/`.

## Data cleaning decisions

- `price_gbp` is converted to numeric.
- `rating` is converted from One–Five to integers 1–5.
- `in_stock` is converted to Boolean.
- Numeric parsing failures are median-imputed.
- Missing category values are represented as `Unknown`.
- `price_inr = price_gbp * 105.50`.

## Database design

`categories` stores unique category names.

`books` stores each book and references its category through `category_id`.

This creates a normalized relationship:

```text
categories (1) ────────< (many) books
```

## Verification

After running all four scripts, check:

```powershell
dir data
dir outputs
```

Then inspect:

- `data/book.csv` — raw scraped data
- `data/book_clean.csv` — cleaned data
- `data/book.db` — SQLite database
- `outputs/query_outputs.txt` — SQL strings and results
- `outputs/read_sql_results.csv` — JOIN recreated with `pd.read_sql`
- `outputs/merge_results.csv` — same JOIN recreated with `pd.merge`

The final line in `query_outputs.txt` reports whether the two JOIN results match.
