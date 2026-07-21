# AI Law Firm — Starter Skeleton

Classification-gated RAG + LangGraph compliance-checking pipeline for
Pakistani legal document review. This is a **starter slice**, not the full
multi-agent system — it implements exactly the flow we designed:

```
User document
     |
     v
[classify_and_retrieve]  <-- embedding routing against category descriptions
     |                        (NOT full-corpus similarity search)
     v
 confident match? --no--> [human_review] --> END
     |yes
     v
  [compare]  <-- strongest LLM, reasons over rule_text + user_document
     |
     v
[human_review] --> END
```

## What's included

- `data/categories/*.json` — 4 verified rule categories (cheque dishonor,
  breach of contract, tenancy eviction, wrongful termination). **Not
  reviewed by a licensed lawyer** — see `review_status` field in each file.
- `app/rag/ingest.py` — loads category JSON into ChromaDB (local, free
  embeddings via sentence-transformers)
- `app/rag/classify.py` — embedding-based routing, returns top-K category
  matches with a confidence threshold (no_confident_match fallback included)
- `app/rag/retrieve.py` — direct metadata fetch by category_id, no search
- `app/main.py` — FastAPI service exposing `/classify`, `/retrieve`,
  `/classify-and-retrieve`
- `app/graph/` — LangGraph pipeline: classify+retrieve node -> conditional
  routing -> compare node (strongest model) -> human review gate

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Ingest the corpus into ChromaDB
python -m app.rag.ingest

# Start the RAG service
uvicorn app.main:app --reload --port 8000

# In another terminal, set your Groq API key and run the graph
export GROQ_API_KEY=your_key_here
python -m app.graph.graph
```

## Next steps (not yet built)

1. **Expand the corpus** — replicate the JSON structure in
   `data/categories/` for more rule categories, following the same
   verified-sourcing process (statute + case law + human review) used
   for the first 4.
2. **Real human-in-the-loop** — `human_review_gate` currently
   auto-approves. Wire it to LangGraph's `interrupt()` so a real person
   reviews before output is returned.
3. **Add the remaining agents** — Intake, Drafting, Summarization,
   Citation Verification — as additional graph nodes once this slice is
   validated end-to-end.
4. **LangSmith tracing** — set `LANGCHAIN_TRACING_V2=true` and your
   LangSmith API key to get visibility into multi-node execution, since
   debugging blind here is much harder than a single chain.
5. **Evaluation set** — build a small set of test documents per category
   with known-correct compliance verdicts to measure the compare_node's
   accuracy, not just eyeball outputs.
6. **Deploy** — same Docker + Hugging Face Spaces pattern as DocNav.
   `Dockerfile` is included and ingests the corpus at build time.

## Design notes / caveats

- Confidence threshold in `classify.py` (0.35) is a starting guess — tune
  it once you have real test documents, since Chroma's raw distance-to-
  confidence conversion is rough.
- `compare_node` uses your strongest model (`llama-3.3-70b-versatile` by
  default) since it's the actual legal reasoning step — keep classification
  cheap/free (embeddings only, no LLM call) per the cost-tiering plan.
- Every rule document's `review_status` field flags that it hasn't been
  reviewed by a licensed lawyer. Don't strip this from the pipeline output
  — surface it to end users too.
