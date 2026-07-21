"""
FastAPI app — this is what you deploy (Docker -> Hugging Face Spaces),
matching your DocNav pattern.

Primary route: /check-document runs the FULL LangGraph pipeline
(intake -> classify -> retrieve -> [compare or contract_redline] ->
[citation_verification if compare] -> deadline -> [drafting if
non-compliant] -> summarization -> human review) in one call, starting
from a raw client message.

The /classify and /retrieve routes are also kept here as thin wrappers
for manual testing/debugging — they are NOT called by the graph itself
(the graph calls classify_document()/retrieve_rule_document() directly,
in-process — see app/graph/nodes.py).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root — must run before ChatGroq() is constructed anywhere

import os
print(f"[startup] GROQ_API_KEY loaded: {bool(os.environ.get('GROQ_API_KEY'))}")
print(f"[startup] SMTP_HOST loaded: {bool(os.environ.get('SMTP_HOST'))}")
print(f"[startup] SMTP_USER loaded: {bool(os.environ.get('SMTP_USER'))}")
print(f"[startup] SMTP_PASSWORD loaded: {bool(os.environ.get('SMTP_PASSWORD'))}")

from app.schemas import ClassifyRequest, ClassifyResponse, RetrieveRequest, RetrieveResponse
from app.rag.classify import classify_document
from app.rag.retrieve import retrieve_rule_document
from app.graph.graph import build_graph

app = FastAPI(
    title="AI Law Firm",
    description="Classification-gated RAG + LangGraph compliance checker.",
    version="0.1.0",
)

# Allows the browser-based frontend (running on a different origin/port,
# e.g. a dev server on :3000/:5173, or a hosted page) to call this API.
# Without this, browsers block the request before it ever reaches the
# routes below and fetch() fails with a generic network error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend URL(s) before deploying publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph once at startup, reuse across requests
law_firm_graph = build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


class CheckDocumentRequest(BaseModel):
    raw_user_message: str
    client_email: str | None = None


@app.post("/check-document")
def check_document(req: CheckDocumentRequest):
    """Main deployed endpoint. See module docstring for the full pipeline.

    Accepts a JSON body: {"raw_user_message": "...", "client_email": "..."}
    (client_email is optional).

    client_email — if provided, notify_node will email a plain-language
    summary + any deadlines found to that address once the pipeline
    finishes (see app/notify/email_sender.py for SMTP setup).

    If intake determines the message is too vague, this returns a
    clarification question instead of a result — the client (or your
    frontend) should re-submit with more detail."""
    result = law_firm_graph.invoke({
        "raw_user_message": req.raw_user_message,
        "client_email": req.client_email,
        "retry_count": 0,
    })

    if result.get("needs_clarification"):
        return {
            "needs_clarification": True,
            "clarification_question": result.get("clarification_question"),
        }

    return {
        "needs_clarification": False,
        "intake_result": result["intake_result"].model_dump() if result.get("intake_result") else None,
        "no_confident_match": result.get("no_confident_match", False),
        "rule_document": result["rule_document"].model_dump() if result.get("rule_document") else None,
        "compliance_result": result["compliance_result"].model_dump() if result.get("compliance_result") else None,
        "redline_result": result["redline_result"].model_dump() if result.get("redline_result") else None,
        "verification_result": result["verification_result"].model_dump() if result.get("verification_result") else None,
        "deadline_result": result["deadline_result"].model_dump() if result.get("deadline_result") else None,
        "drafted_document": result["drafted_document"].model_dump() if result.get("drafted_document") else None,
        "client_summary": result["client_summary"].model_dump() if result.get("client_summary") else None,
        "notification_sent": result.get("notification_sent", False),
    }


# --- Manual-testing-only routes below (graph does not call these) ---

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.document_text.strip():
        raise HTTPException(status_code=400, detail="document_text cannot be empty")
    return classify_document(req.document_text)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    return retrieve_rule_document(req.category_id)