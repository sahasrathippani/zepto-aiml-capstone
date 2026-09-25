from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data") / "book.csv"
OUTPUT_FILE = Path("data") / "book_clean.csv"

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def clean_books():
    df = pd.read_csv(INPUT_FILE)

    # Clean price and record parsing failures.
    df["price_gbp"] = (
        df["price_gbp"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.strip()
    )
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")

    # Clean star rating to integer 1-5.
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Convert availability text to boolean.
    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains("In stock", case=False, na=False)
    )

    # Median-impute numeric parsing failures as required.
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median())

    df["rating"] = df["rating"].round().astype(int).clip(1, 5)

    # Missing categories are retained as an explicit category.
    df["category"] = df["category"].fillna("Unknown").replace("", "Unknown")

    # Fixed project conversion rate: 1 GBP = 105.50 INR.
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    # Keep the required cleaned fields.
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned data to {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    clean_books()
