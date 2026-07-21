"""
Scoped retrieval: given a category_id (already decided by classify.py),
fetch ONLY that category's full rule document. This is a direct lookup,
not a similarity search — no other category's chunks are ever touched,
which is the whole point of classification-gated retrieval.
"""
import json

from app.rag.ingest import get_chroma_client, COLLECTION_NAME
from chromadb.utils import embedding_functions
from app.schemas import RetrieveResponse, RuleDocument


def _get_collection():
    client = get_chroma_client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_rule_document(category_id: str) -> RetrieveResponse:
    collection = _get_collection()

    result = collection.get(ids=[category_id])

    if not result["ids"]:
        return RetrieveResponse(found=False, rule_document=None)

    meta = result["metadatas"][0]
    full_rule = json.loads(meta["full_rule_json"])

    rule_doc = RuleDocument(**full_rule)
    return RetrieveResponse(found=True, rule_document=rule_doc)


if __name__ == "__main__":
    resp = retrieve_rule_document("PPC-489F-CHEQUE-DISHONOR")
    print(resp.model_dump_json(indent=2))