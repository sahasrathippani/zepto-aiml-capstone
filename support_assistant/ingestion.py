from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, EMBEDDING_MODEL


def load_documents():
    documents = []
    ids = []
    metadatas = []

    for path in sorted(Path(DOCS_DIR).glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # One chunk per document is acceptable for the supplied document lengths.
        documents.append(text)
        ids.append(path.stem)
        metadatas.append({"source": path.name})

    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(documents)}")

    return ids, documents, metadatas


def build_collection(reset=True):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = load_documents()

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


if __name__ == "__main__":
    collection = build_collection()
    print("Chroma collection:", COLLECTION_NAME)
    print("Documents indexed:", collection.count())
