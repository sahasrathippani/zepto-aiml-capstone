from pathlib import Path
import sqlite3

import pandas as pd

INPUT_FILE = Path("data") / "book_clean.csv"
DB_FILE = Path("data") / "book.db"


def create_database():
    df = pd.read_csv(INPUT_FILE)

    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category_id INTEGER NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )

        categories = sorted(df["category"].dropna().unique())

        for category in categories:
            conn.execute(
                "INSERT INTO categories (category_name) VALUES (?)",
                (category,),
            )

        category_map = dict(
            conn.execute(
                "SELECT category_name, category_id FROM categories"
            ).fetchall()
        )

        for row in df.itertuples(index=False):
            conn.execute(
                """
                INSERT INTO books
                (title, category_id, price_gbp, price_inr, rating, in_stock)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row.title,
                    category_map[row.category],
                    float(row.price_gbp),
                    float(row.price_inr),
                    int(row.rating),
                    int(bool(row.in_stock)),
                ),
            )

        conn.commit()

        book_count = conn.execute(
            "SELECT COUNT(*) FROM books"
        ).fetchone()[0]
        category_count = conn.execute(
            "SELECT COUNT(*) FROM categories"
        ).fetchone()[0]

    print(f"Database created: {DB_FILE}")
    print(f"Books inserted: {book_count}")
    print(f"Categories inserted: {category_count}")


if __name__ == "__main__":
    create_database()
