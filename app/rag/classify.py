"""
Classification-gated routing.

This is deliberately NOT a full-corpus similarity search. It only ever
compares the user's document against the small set of category
descriptions (~4 today, ~20-40 once you expand the corpus) — cheap and fast,
per the rate-limit design discussed for this project.

A confidence threshold decides whether we trust the match or flag
"no_confident_match" so the graph can route to human review / general RAG
fallback instead of silently guessing.
"""
import json

from app.rag.ingest import get_chroma_client, COLLECTION_NAME
from chromadb.utils import embedding_functions
from app.schemas import ClassifyResponse, CategoryMatch

# With cosine distance (range 0-2, where 0 = identical), 0.45 is a
# reasonable starting threshold for short description matching. Tune this
# once you have a real labeled test set — don't treat this as final.
CONFIDENCE_THRESHOLD = 0.45
TOP_K = 3


def _get_collection():
    client = get_chroma_client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def classify_document(document_text: str) -> ClassifyResponse:
    collection = _get_collection()

    results = collection.query(
        query_texts=[document_text],
        n_results=TOP_K,
    )

    matches: list[CategoryMatch] = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for cat_id, distance, meta in zip(ids, distances, metadatas):
        # Cosine distance ranges 0 (identical) to 2 (opposite). Convert to a
        # 0-1 confidence score: confidence = 1 - (distance / 2).
        confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        matches.append(CategoryMatch(
            category_id=cat_id,
            title=meta["title"],
            confidence=round(confidence, 3),
            reasoning=f"Embedding similarity match on category description (distance={distance:.3f})",
        ))

    matches.sort(key=lambda m: m.confidence, reverse=True)
    top = matches[0] if matches else None
    no_confident_match = top is None or top.confidence < CONFIDENCE_THRESHOLD

    return ClassifyResponse(
        matches=matches,
        top_match=top,
        no_confident_match=no_confident_match,
    )


if __name__ == "__main__":
    sample = (
        "The tenant has not paid rent for the last two months and refuses "
        "to vacate the rented apartment despite written notice."
    )
    result = classify_document(sample)
    print(json.dumps(result.model_dump(), indent=2))