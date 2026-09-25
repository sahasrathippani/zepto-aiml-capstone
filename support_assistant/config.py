import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "zepto_policy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default is mock mode. Set MOCK_LLM=0 only for the optional real-LLM extension.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
