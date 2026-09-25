import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
OUTPUT_FILE = Path("data") / "book.csv"
PAGES_TO_SCRAPE = 5


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def rating_to_text(class_list):
    for rating in ["One", "Two", "Three", "Four", "Five"]:
        if rating in class_list:
            return rating
    return ""


def scrape_books():
    rows = []

    for page_number in range(1, PAGES_TO_SCRAPE + 1):
        page_url = (
            BASE_URL
            if page_number == 1
            else f"{BASE_URL}catalogue/page-{page_number}.html"
        )

        print(f"Scraping page {page_number}...")
        soup = get_soup(page_url)

        books = soup.select("article.product_pod")

        for book in books:
            title_tag = book.select_one("h3 a")
            price_tag = book.select_one(".price_color")
            availability_tag = book.select_one(".availability")
            rating_tag = book.select_one("p.star-rating")

            title = title_tag.get("title", "").strip()
            price = price_tag.get_text(strip=True) if price_tag else ""
            availability = (
                availability_tag.get_text(" ", strip=True)
                if availability_tag
                else ""
            )

            rating = (
                rating_to_text(rating_tag.get("class", []))
                if rating_tag
                else ""
            )

            relative_book_url = title_tag.get("href", "")
            book_url = requests.compat.urljoin(page_url, relative_book_url)

            # Category is available on the individual book page.
            detail_soup = get_soup(book_url)
            breadcrumb = detail_soup.select("ul.breadcrumb li")

            category = ""
            if len(breadcrumb) >= 3:
                category = breadcrumb[2].get_text(strip=True)

            rows.append(
                {
                    "title": title,
                    "price_gbp": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category,
                }
            )

            time.sleep(0.05)

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} books to {OUTPUT_FILE}")
    print(f"Categories found: {df['category'].nunique()}")
    print(df.head())


if __name__ == "__main__":
    scrape_books()
