from pathlib import Path
import sqlite3

import pandas as pd

DB_FILE = Path("data") / "book.db"
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "query_outputs.txt"
READ_SQL_FILE = OUTPUT_DIR / "read_sql_results.csv"
MERGE_FILE = OUTPUT_DIR / "merge_results.csv"

QUERIES = {
    "Q1_SELECT_WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, price_gbp DESC;
    """,
    "Q2_ORDER_BY_LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "Q3_DISTINCT": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "Q4_BETWEEN": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,
    "Q5_IN": """
        SELECT title, category_id, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC;
    """,
    "Q6_JOIN": """
        SELECT
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title;
    """,
}


def run_queries():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for name, sql in QUERIES.items():
                f.write("=" * 80 + "\n")
                f.write(f"{name}\n")
                f.write("=" * 80 + "\n")
                f.write(sql.strip() + "\n\n")

                result = pd.read_sql(sql, conn)
                f.write(result.to_string(index=False))
                f.write("\n\n")

        # Requirement: reproduce a JOIN result using pd.read_sql.
        join_sql = QUERIES["Q6_JOIN"]
        read_sql_join = pd.read_sql(join_sql, conn)
        read_sql_join.to_csv(READ_SQL_FILE, index=False)

        # Reproduce the same JOIN using pd.merge.
        books_df = pd.read_sql(
            """
            SELECT
                book_id, title, category_id,
                price_gbp, price_inr, rating, in_stock
            FROM books
            """,
            conn,
        )

        categories_df = pd.read_sql(
            """
            SELECT category_id, category_name
            FROM categories
            """,
            conn,
        )

        merge_result = books_df.merge(
            categories_df,
            on="category_id",
            how="inner",
        )

        merge_result = merge_result[
            [
                "book_id",
                "title",
                "category_name",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
            ]
        ].sort_values(["category_name", "title"]).reset_index(drop=True)

        read_sql_check = read_sql_join.copy()
        read_sql_check = read_sql_check[
            [
                "book_id",
                "title",
                "category_name",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
            ]
        ].sort_values(["category_name", "title"]).reset_index(drop=True)

        matches = read_sql_check.equals(merge_result)

        merge_result.to_csv(MERGE_FILE, index=False)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("pd.read_sql vs pd.merge verification\n")
            f.write("=" * 80 + "\n")
            f.write(f"Results match: {matches}\n")

    print(f"Saved query strings and outputs to: {OUTPUT_FILE}")
    print(f"Saved pd.read_sql JOIN result to: {READ_SQL_FILE}")
    print(f"Saved pd.merge result to: {MERGE_FILE}")
    print(f"JOIN results match: {matches}")


if __name__ == "__main__":
    run_queries()
