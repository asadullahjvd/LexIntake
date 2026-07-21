"""
Pydantic schemas shared across the RAG service and LangGraph agents.
Keep every cross-node payload defined here so the graph's state stays typed.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CategoryMatch(BaseModel):
    """One candidate category returned by the classifier."""
    category_id: str
    title: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ClassifyRequest(BaseModel):
    document_text: str


class ClassifyResponse(BaseModel):
    matches: list[CategoryMatch]
    top_match: Optional[CategoryMatch] = None
    no_confident_match: bool = False


class RuleDocument(BaseModel):
    """The full 2-page rule reference for one category — this is what
    the retrieval step returns and what the comparison agent consumes."""
    category_id: str
    title: str
    jurisdiction: str
    legal_area: str
    statute_text: str
    elements_required: list[str]
    common_defenses: list[str]
    procedural_notes: list[str]
    example_scenario: str
    sources: list[str]
    review_status: str


class RetrieveRequest(BaseModel):
    category_id: str


class RetrieveResponse(BaseModel):
    found: bool
    rule_document: Optional[RuleDocument] = None


class ComplianceViolation(BaseModel):
    element_or_rule: str
    issue: str
    severity: Severity


class ComplianceResult(BaseModel):
    """Structured output of the Compare/Compliance agent (Node 3)."""
    category_id: str
    compliant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    violations: list[ComplianceViolation]
    missing_elements: list[str]
    summary: str
    disclaimer: str = (
        "AI-assisted legal reference output. Not verified legal advice. "
        "Consult a licensed lawyer before relying on this result."
    )


class VerificationIssue(BaseModel):
    """One specific claim from ComplianceResult that failed verification."""
    claim: str = Field(description="The exact claim being checked, quoted or closely paraphrased from the compliance result")
    problem: str = Field(description="Why this claim isn't supported — e.g. 'not stated in rule text' or 'not present in user document'")


class VerificationResult(BaseModel):
    """Structured output of the Citation Verification agent (Node 4).

    This agent's ONLY job is to check that compare_node's claims are
    actually grounded in the rule_document and user_document_text it was
    given — not invented. It does not re-do the compliance analysis itself."""
    verified: bool
    issues: list[VerificationIssue]
    notes: str = Field(description="Brief overall verification note, e.g. 'all claims grounded' or summary of what's wrong")


class DocumentType(str, Enum):
    """What kind of thing the client submitted — determines which agents
    actually run (see route_after_classification / route_after_deadline
    in nodes.py). Not every document type needs every agent: a court
    notice doesn't need compliance-checking, a judgment doesn't need
    drafting by default, etc."""
    DISPUTE_NARRATIVE = "dispute_narrative"   # client describing a situation in their own words
    CONTRACT_FOR_REVIEW = "contract_for_review"  # a contract/agreement pasted in for risk review
    COURT_NOTICE = "court_notice"             # official notice/summons from a court
    LEGAL_NOTICE = "legal_notice"             # formal notice from the other party (e.g. notice to vacate, demand notice)
    JUDGMENT = "judgment"                     # a court judgment/order/ruling
    PETITION = "petition"                     # a petition filed (by either party)
    COMPLAINT = "complaint"                   # a formal complaint filed (by either party)


class IntakeResult(BaseModel):
    """Structured output of the Intake / Client Interview agent (Node 0).

    Turns a raw, possibly incomplete user message into structured case
    facts. If critical information is missing, ready_for_classification
    is False and follow_up_question asks for exactly what's needed —
    the graph should stop and surface that question rather than guessing."""
    parties: list[str] = Field(default_factory=list, description="Named parties involved, e.g. ['tenant: Ms. Fatima', 'landlord: Mr. Riaz']")
    client_role: str = Field(description="The role of the person asking for help (the CLIENT), e.g. 'tenant', 'landlord', 'employee', 'employer', 'buyer', 'seller'. Determine this from who is telling the story / seeking help — the narrator of the message is almost always the client.")
    key_dates: list[str] = Field(default_factory=list, description="Any dates mentioned, e.g. ['rent last paid March 2026']")
    jurisdiction: Optional[str] = Field(default=None, description="Province/city if mentioned, e.g. 'Punjab' or 'Lahore'")
    issue_description: str = Field(description="Plain-language summary of the legal issue")
    desired_outcome: Optional[str] = Field(default=None, description="What the client wants, if stated")
    ready_for_classification: bool = Field(description="True only if enough info exists to proceed to classification")
    follow_up_question: Optional[str] = Field(default=None, description="ONE specific question to ask if ready_for_classification is False")
    structured_document_text: str = Field(description="Clean, structured synthesis of the case facts, to be used as the document text for classification/retrieval/comparison")
    document_type: DocumentType = Field(description="What kind of thing the client submitted — determines which agents run downstream")


class DraftedDocument(BaseModel):
    """Structured output of the Document Drafting agent (Node 5).

    Only produced when compliance_result.compliant is False — this drafts
    a document that serves the CLIENT's interests (see client_role in
    IntakeResult), not generically "the document associated with this rule
    category". A rule's compliance checklist is often written from one
    party's perspective (e.g. landlord obligations); the draft must serve
    whichever party is actually the client, even if that means drafting a
    response/objection rather than fulfilling the other party's paperwork."""
    document_type: str = Field(description="What kind of document this is, e.g. 'Tenant Response to Improper Eviction', 'Demand Letter'")
    title: str
    body: str = Field(description="The full drafted document text")
    drafted_for_role: str = Field(description="Which party this document serves — should match the client_role from intake")
    addresses_gaps: list[str] = Field(description="Which specific violations/missing_elements this draft responds to or leverages on the client's behalf")
    drafting_notes: list[str] = Field(default_factory=list, description="Things the client should fill in manually, e.g. '[INSERT EXACT DATE]'")
    disclaimer: str = (
        "AI-generated draft. Not a substitute for review by a licensed lawyer "
        "before use in any legal or official capacity."
    )


class ClientSummary(BaseModel):
    """Structured output of the Summarization agent (Node 6).

    Distinct from ComplianceResult.summary (which is written alongside the
    technical legal analysis, for a technical/downstream-agent audience).
    This is specifically for the CLIENT to read — plain language, no
    legalese, empathetic but factual, with concrete next steps."""
    plain_summary: str = Field(description="3-5 sentences in plain language explaining the situation and outcome — no legal jargon, no statute citations")
    key_takeaway: str = Field(description="One sentence: the single most important thing the client needs to know")
    next_steps: list[str] = Field(description="Concrete, actionable next steps for the client, in plain language")
    disclaimer: str = (
        "This is a plain-language summary generated by AI and is not legal "
        "advice. Please consult a licensed lawyer before taking action."
    )


class RedlineIssue(BaseModel):
    """One flagged clause-level risk in a contract under review."""
    clause_excerpt: str = Field(description="The specific clause or phrase being flagged, quoted or closely paraphrased from the contract")
    risk_level: Severity
    issue: str = Field(description="What's risky, missing, or non-standard about this clause")
    suggested_redline: str = Field(description="Specific suggested replacement or addition text to fix the issue")


class RedlineResult(BaseModel):
    """Structured output of the Contract Review / Redline agent.

    Different job from ComplianceResult: instead of checking a client's
    SITUATION against one rule (did X happen, was Y followed), this scans
    a full CONTRACT DOCUMENT clause-by-clause against the rule reference's
    known requirements/defenses/procedural notes, flagging risky, missing,
    or non-standard terms — a comparison/analysis task, not generation."""
    category_id: str
    overall_risk: Severity = Field(description="Overall risk level of the contract as a whole")
    issues: list[RedlineIssue]
    missing_protections: list[str] = Field(description="Standard protections this contract type would normally include but doesn't")
    summary: str
    disclaimer: str = (
        "AI-assisted contract review. Not a substitute for review by a "
        "licensed lawyer before signing or relying on this contract."
    )


class DeadlineItem(BaseModel):
    """One procedural deadline or limitation period relevant to the case."""
    deadline_type: str = Field(description="What this deadline is for, e.g. 'Notice response window', 'Limitation period to file suit'")
    description: str = Field(description="Plain explanation of what must happen and why")
    days_from_reference: Optional[int] = Field(default=None, description="Number of days allowed, if the rule specifies a fixed period")
    reference_event: str = Field(description="What event the countdown starts from, e.g. 'date the notice was received', 'date rent became due'")
    source: str = Field(description="Where this deadline comes from in the rule reference, e.g. 'procedural_notes' or 'elements_required'")


class DeadlineResult(BaseModel):
    """Structured output of the Deadline / Calendar agent (Node 7).

    Extracts procedural deadlines and limitation periods from the rule
    reference relevant to this case. Does NOT calculate actual calendar
    dates — the LLM is unreliable at date arithmetic and the client's
    exact reference dates are often imprecise in the narrative. Instead
    this surfaces the RULES (days_from_reference + reference_event) so
    the client or a lawyer can compute the actual deadline accurately."""
    deadlines: list[DeadlineItem]
    summary: str
    disclaimer: str = (
        "These are general deadline rules extracted from the applicable "
        "law, not calculated calendar dates. Exact deadlines depend on "
        "precise dates in your case — confirm with a licensed lawyer."
    )