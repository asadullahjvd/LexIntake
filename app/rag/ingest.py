"""
Ingests every category JSON in data/categories/ into ChromaDB.

Two things get stored per category:
1. A short "category description" embedding — used ONLY for classification routing
   (comparing the user's document against ~20-40 category summaries, not full text).
2. The full rule document as metadata, fetched later by category_id — NOT re-searched.

Run: python -m app.rag.ingest
"""
import json
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "categories")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_store")

COLLECTION_NAME = "law_categories"


def get_chroma_client():
    """Shared client config — telemetry disabled to silence the harmless but
    noisy 'Failed to send telemetry event' errors caused by a chromadb/posthog
    version mismatch."""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def build_category_description(cat: dict) -> str:
    """Short text used purely for routing/classification — NOT the full rule text.
    Keep this focused on 'what kind of case does this cover' so classification
    matches on situation type, not on legal minutiae."""
    return (
        f"{cat['title']}. Legal area: {cat['legal_area']}. "
        f"Keywords: {', '.join(cat['keywords'])}. "
        f"Applies when: {cat['example_scenario']}"
    )


def ingest():
    client = get_chroma_client()

    # Free local embedding model — swap for a hosted one later if needed
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # IMPORTANT: explicitly use cosine distance. ChromaDB defaults to raw L2
    # distance, which is NOT bounded 0-1 and breaks any "confidence = 1 - distance"
    # calculation downstream (see classify.py). Cosine distance IS bounded and
    # matches what "confidence" should mean here.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    category_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    if not category_files:
        print(f"No category JSON files found in {DATA_DIR}")
        return

    ids, documents, metadatas = [], [], []

    for filename in category_files:
        with open(os.path.join(DATA_DIR, filename), "r") as f:
            cat = json.load(f)

        description = build_category_description(cat)

        ids.append(cat["category_id"])
        documents.append(description)
        metadatas.append({
            "category_id": cat["category_id"],
            "title": cat["title"],
            "jurisdiction": cat["jurisdiction"],
            "legal_area": cat["legal_area"],
            "full_rule_json": json.dumps(cat),
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingested {len(ids)} categories into '{COLLECTION_NAME}': {ids}")


if __name__ == "__main__":
    ingest()