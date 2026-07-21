"""
Node implementations for the AI Law Firm LangGraph pipeline.

Model tiering (per the cost/rate-limit design discussed):
  - intake_node: cheap/fast model — extraction and light judgment, not deep legal reasoning
  - classify_and_retrieve_node: no LLM call at all — embedding similarity + direct fetch (cheapest)
  - compare_node: your STRONGEST model — this is the actual legal reasoning step
  - citation_verification_node: also your strongest model — needs careful reading, not speed
  - drafting_node: your strongest model — drafting requires the same care as compliance reasoning,
    and must be perspective-aware (see client_role) to avoid drafting against the client's own interests
  - contract_redline_node: your strongest model — clause risk assessment needs the same rigor
  - deadline_node: cheap/fast model — extracting deadline RULES from the rule text, not calculating
    dates (which LLMs are unreliable at) or doing new legal reasoning
  - summarization_node: cheap/fast model — tone/format transformation of already-completed analysis,
    not new legal reasoning

RAG functions are called DIRECTLY (in-process), not over HTTP — no need to
run main.py/uvicorn separately just to execute the graph. main.py still
exists and wraps this same graph for when you actually deploy it (see README).
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.graph.state import LawFirmState
from app.schemas import (
    ComplianceResult, RuleDocument, VerificationResult, IntakeResult,
    DraftedDocument, ClientSummary, RedlineResult, DeadlineResult, DocumentType,
)
from app.rag.classify import classify_document
from app.notify.email_sender import send_email
from dateparser.search import search_dates
from app.rag.retrieve import retrieve_rule_document

# Tiering: cheap/fast model for lightweight tasks, strong model for the
# actual compliance reasoning. Adjust model names to what's current on Groq.
MODEL_STRONG = "llama-3.3-70b-versatile"
MODEL_CHEAP = "llama-3.1-8b-instant"

# Hard cap on compare <-> verify retries, per the iteration-guard design —
# never let a bad verification loop forever and burn API calls.
MAX_VERIFICATION_RETRIES = 2


# ---------------------------------------------------------------------------
# Node 0: Intake / Client Interview — turns a raw message into case facts
# ---------------------------------------------------------------------------

INTAKE_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal intake assistant. Read the client's message and extract
structured case facts. Do NOT give legal advice or conclusions — only
extract and organize what the client has told you.

IMPORTANT: identify client_role — the role of the PERSON SEEKING HELP (not
just any party mentioned). The narrator of the message — the "I" — is
almost always the client. For example, if the message describes "my
landlord did X to me", the client_role is "tenant", NOT "landlord", even
though landlord is also a party in the story. Getting this wrong means
downstream documents could be drafted against the client's own interests.

IMPORTANT: also identify document_type — what kind of thing the client
actually submitted. This determines which downstream steps run, so get it
right rather than defaulting to dispute_narrative:
- "dispute_narrative" — the client is describing a situation or problem
  in their own words (e.g. "my landlord kicked me out", "I wasn't paid")
- "contract_for_review" — the client pasted an actual contract/agreement's
  terms and wants it reviewed for risk, NOT a description of a dispute
- "court_notice" — the client received an official notice/summons FROM A
  COURT (e.g. a hearing date notice, a summons to appear)
- "legal_notice" — the client received a formal notice FROM THE OTHER
  PARTY (not a court) — e.g. a notice to vacate, a legal demand letter
- "judgment" — the client has an actual court judgment, order, or ruling
  and wants to understand it (not a notice — a decided outcome)
- "petition" — the client has a petition that was filed (by either side)
- "complaint" — the client has a formal complaint that was filed (by
  either side) — distinct from a dispute_narrative, which is the client's
  own informal description, not an actual filed complaint document

If the client only describes their situation in plain words with no
actual document/notice/filing mentioned or pasted, that's
dispute_narrative — don't classify a plain description as one of the
document types above just because it mentions a notice was received;
only use those types when the client is presenting the actual document's
content, not just narrating that something happened.

Decide if there's enough information to proceed:
- Set ready_for_classification=True if the message describes a clear
  situation (who is involved, roughly what happened) even if some minor
  details are missing.
- Set ready_for_classification=False ONLY if the message is too vague to
  even identify the type of legal issue (e.g. "I have a legal problem,
  help me") — in that case, ask exactly ONE specific follow_up_question
  to get the minimum information needed to proceed.

If ready_for_classification is True, also write structured_document_text:
a clean, well-organized paragraph synthesizing the case facts, suitable
for a legal classification system to read (this replaces the client's
raw, possibly messy message).

CLIENT MESSAGE:
{raw_user_message}

{format_instructions}
"""
)


def intake_node(state: LawFirmState) -> LawFirmState:
    """Cheap/fast model — this is extraction and light judgment, not deep
    legal reasoning, so it doesn't need the strong tier."""
    parser = PydanticOutputParser(pydantic_object=IntakeResult)
    llm = ChatGroq(model=MODEL_CHEAP, temperature=0)
    chain = INTAKE_PROMPT | llm | parser

    result: IntakeResult = chain.invoke({
        "raw_user_message": state["raw_user_message"],
        "format_instructions": parser.get_format_instructions(),
    })

    if not result.ready_for_classification:
        return {
            **state,
            "intake_result": result,
            "needs_clarification": True,
            "clarification_question": result.follow_up_question,
        }

    return {
        **state,
        "intake_result": result,
        "needs_clarification": False,
        "user_document_text": result.structured_document_text,
    }


def route_after_intake(state: LawFirmState) -> str:
    if state.get("needs_clarification"):
        return "end_for_clarification"
    return "classify_and_retrieve"


# ---------------------------------------------------------------------------
# Node 1: Classify + Node 2: Retrieve (combined — direct function calls)
# ---------------------------------------------------------------------------

def classify_and_retrieve_node(state: LawFirmState) -> LawFirmState:
    """Calls classify_document() and retrieve_rule_document() directly —
    no network call. Cheap and fast: only embedding similarity + a metadata
    lookup happen here, no LLM involved."""
    classification = classify_document(state["user_document_text"])

    if classification.no_confident_match or classification.top_match is None:
        return {
            **state,
            "classification": classification,
            "no_confident_match": True,
            "requires_human_review": True,
        }

    retrieval = retrieve_rule_document(classification.top_match.category_id)

    if not retrieval.found:
        return {
            **state,
            "classification": classification,
            "no_confident_match": True,
            "requires_human_review": True,
        }

    return {
        **state,
        "classification": classification,
        "rule_document": retrieval.rule_document,
        "no_confident_match": False,
    }


def route_after_classification(state: LawFirmState) -> str:
    """Branches on document_type (set by intake_node) once a rule category
    is known. Only the agents actually relevant to this document type run:

    - contract_for_review -> contract_redline (clause-by-clause risk scan)
    - dispute_narrative   -> compare (compliance check against the rule)
    - court_notice, legal_notice, petition, complaint, judgment ->
      straight to "deadline", SKIPPING compare/citation_verification/redline
      entirely. These are documents the client RECEIVED or that were
      already FILED — there's no "compliance check" to run on them; the
      client needs to understand deadlines and (for notices/petitions/
      complaints) get a response drafted, not have their situation
      checked against a rule.

    No confident category match sends any path straight to human review."""
    if state.get("no_confident_match"):
        return "human_review"

    intake_result = state.get("intake_result")
    document_type = intake_result.document_type if intake_result else DocumentType.DISPUTE_NARRATIVE

    if document_type == DocumentType.CONTRACT_FOR_REVIEW:
        return "contract_redline"

    if document_type == DocumentType.DISPUTE_NARRATIVE:
        return "compare"

    # court_notice, legal_notice, judgment, petition, complaint all skip
    # straight to deadline extraction — none of them need compliance
    # checking, citation verification, or contract redlining.
    return "deadline"


# ---------------------------------------------------------------------------
# Node 3b: Contract Review / Redline — clause-by-clause risk scan
# ---------------------------------------------------------------------------

REDLINE_PROMPT = ChatPromptTemplate.from_template(
    """You are a contract reviewer. Scan the contract below clause by clause
against the rule reference. Flag clauses that are risky, non-standard, or
missing protections a contract of this type would normally include. This is
a RISK SCAN across the whole document, not a check for one specific
violation — look for multiple independent issues if they exist.

Do NOT invent clauses that aren't in the contract. If a standard protection
is simply absent, list it in missing_protections rather than fabricating a
clause_excerpt for something that isn't there.

RULE REFERENCE - {title} ({category_id})
Statute: {statute_text}
Required elements: {elements_required}
Common defenses: {common_defenses}
Procedural notes: {procedural_notes}

CONTRACT TO REVIEW:
{user_document_text}

For each risky/non-standard clause found, provide the clause excerpt, risk
level, what's wrong, and a specific suggested redline. Also list any
standard protections this contract type would normally include but doesn't.
Give an overall_risk rating for the contract as a whole.

{format_instructions}
"""
)


def contract_redline_node(state: LawFirmState) -> LawFirmState:
    """Only reached when intake_result.document_type is 'contract_for_review'
    (see route_after_classification). Uses the strong model tier — clause
    risk assessment needs the same rigor as compliance reasoning. This is a
    separate path from compare_node/compliance_result: a contract redline
    scans a whole document for many possible issues, while compare_node
    checks one narrative against one rule's requirements."""
    rule_doc = state["rule_document"]
    if rule_doc is None:
        raise ValueError("contract_redline_node reached with no rule_document — check graph routing")

    parser = PydanticOutputParser(pydantic_object=RedlineResult)
    llm = ChatGroq(model=MODEL_STRONG, temperature=0)
    chain = REDLINE_PROMPT | llm | parser

    result: RedlineResult = chain.invoke({
        "title": rule_doc.title,
        "category_id": rule_doc.category_id,
        "statute_text": rule_doc.statute_text,
        "elements_required": rule_doc.elements_required,
        "common_defenses": rule_doc.common_defenses,
        "procedural_notes": rule_doc.procedural_notes,
        "user_document_text": state["user_document_text"],
        "format_instructions": parser.get_format_instructions(),
    })

    return {**state, "redline_result": result}


# ---------------------------------------------------------------------------
# Node 3: Compare / Compliance check — the real reasoning step
# ---------------------------------------------------------------------------

COMPARE_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal compliance checker. Compare the user's document against
the rule reference below. Do NOT invent facts not present in either document.
If uncertain, reflect that uncertainty in your confidence score rather than
guessing.

RULE REFERENCE - {title} ({category_id})
Statute: {statute_text}
Required elements: {elements_required}
Common defenses: {common_defenses}

USER'S DOCUMENT:
{user_document_text}
{feedback_section}
Return your analysis as structured output: whether the document appears
compliant, any violations found (with severity), any missing required
elements, and a plain-language summary.

{format_instructions}
"""
)


def compare_node(state: LawFirmState) -> LawFirmState:
    rule_doc = state["rule_document"]
    if rule_doc is None:
        raise ValueError("compare_node reached with no rule_document — check graph routing")

    parser = PydanticOutputParser(pydantic_object=ComplianceResult)
    llm = ChatGroq(model=MODEL_STRONG, temperature=0)

    chain = COMPARE_PROMPT | llm | parser

    # If citation_verification_node sent this back for a retry, include its
    # feedback so this attempt actually corrects the flagged issues instead
    # of repeating the same mistake.
    feedback = state.get("verification_feedback")
    feedback_section = (
        f"\nA previous attempt at this analysis had issues that were NOT "
        f"supported by the rule text or user document:\n{feedback}\n"
        f"Correct these specific issues in this attempt.\n"
        if feedback else "\n"
    )

    result: ComplianceResult = chain.invoke({
        "title": rule_doc.title,
        "category_id": rule_doc.category_id,
        "statute_text": rule_doc.statute_text,
        "elements_required": rule_doc.elements_required,
        "common_defenses": rule_doc.common_defenses,
        "user_document_text": state["user_document_text"],
        "feedback_section": feedback_section,
        "format_instructions": parser.get_format_instructions(),
    })

    # Clear feedback now that it's been used, so it doesn't leak into a
    # future unrelated run if state gets reused.
    return {**state, "compliance_result": result, "verification_feedback": None}


# ---------------------------------------------------------------------------
# Node 4: Citation Verification — checks compare_node's claims are grounded
# ---------------------------------------------------------------------------

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict fact-checker. Your ONLY job is to verify that every
claim in the COMPLIANCE RESULT below is actually supported by the RULE
REFERENCE and the USER'S DOCUMENT. You are NOT re-doing the compliance
analysis — you are checking whether the analysis's claims are grounded in
its sources or invented.

For each violation and each missing_element in the compliance result, check:
1. Does it reference something that actually appears in the rule reference
   (statute text, elements_required, or common_defenses)? If the compliance
   result claims a rule requirement that isn't in the rule reference, flag it.
2. Does it reference something that actually appears in (or is genuinely
   absent from) the user's document? If it claims the user's document says
   something it doesn't say, flag it.

Be strict — a claim that is a reasonable inference from the text is fine;
a claim that references specifics not present in either source is not.

RULE REFERENCE - {title} ({category_id})
Statute: {statute_text}
Required elements: {elements_required}
Common defenses: {common_defenses}

USER'S DOCUMENT:
{user_document_text}

COMPLIANCE RESULT TO VERIFY:
Compliant: {compliant}
Violations: {violations}
Missing elements: {missing_elements}
Summary: {summary}

Return structured output: verified=true only if ALL claims are grounded,
listing any unsupported claims as issues with a brief explanation each.

{format_instructions}
"""
)


def citation_verification_node(state: LawFirmState) -> LawFirmState:
    """Independently re-checks compare_node's output against its sources.
    Uses the same strong model tier as compare_node — this step needs
    careful reading, not speed. If verification fails and retries remain,
    sends the case back to compare_node with specific feedback; otherwise
    flags it for human review regardless of the compliance verdict."""
    rule_doc = state["rule_document"]
    compliance_result = state["compliance_result"]

    if rule_doc is None or compliance_result is None:
        raise ValueError(
            "citation_verification_node reached without rule_document or "
            "compliance_result — check graph routing"
        )

    parser = PydanticOutputParser(pydantic_object=VerificationResult)
    llm = ChatGroq(model=MODEL_STRONG, temperature=0)
    chain = VERIFY_PROMPT | llm | parser

    result: VerificationResult = chain.invoke({
        "title": rule_doc.title,
        "category_id": rule_doc.category_id,
        "statute_text": rule_doc.statute_text,
        "elements_required": rule_doc.elements_required,
        "common_defenses": rule_doc.common_defenses,
        "user_document_text": state["user_document_text"],
        "compliant": compliance_result.compliant,
        "violations": [v.model_dump() for v in compliance_result.violations],
        "missing_elements": compliance_result.missing_elements,
        "summary": compliance_result.summary,
        "format_instructions": parser.get_format_instructions(),
    })

    retry_count = state.get("retry_count", 0)

    if result.verified:
        return {**state, "verification_result": result}

    if retry_count < MAX_VERIFICATION_RETRIES:
        feedback_text = "; ".join(f"{i.claim} — {i.problem}" for i in result.issues)
        return {
            **state,
            "verification_result": result,
            "verification_feedback": feedback_text,
            "retry_count": retry_count + 1,
        }

    return {
        **state,
        "verification_result": result,
        "requires_human_review": True,
    }


def route_after_verification(state: LawFirmState) -> str:
    """Two outcomes now (compliant/not branching moved to route_after_deadline):
    1. Verification failed and retries remain -> back to compare_node
    2. Verified (or retries exhausted) -> proceed to deadline_node, which
       runs regardless of compliant/not before the drafting/summarization split"""
    result = state.get("verification_result")

    if result is not None and not result.verified and state.get("verification_feedback"):
        return "compare"

    return "deadline"


# ---------------------------------------------------------------------------
# Node 5: Document Drafting — drafts the fix, only when non-compliant
# ---------------------------------------------------------------------------

DRAFTING_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal document drafter. The compliance check below found
that the document is NOT fully compliant with the rule reference.

CRITICAL: the CLIENT's role is "{client_role}". You must draft a document
that SERVES AND PROTECTS the client's interests as the {client_role} — you
must NOT draft a document that would be used by the other party against
the client, even if the compliance gaps found are technically framed as
that other party's obligations under the rule.

For example: if the client is a tenant and the compliance gaps are things
like "landlord did not issue proper written notice" or "landlord did not
file with the Rent Tribunal", do NOT draft that missing landlord paperwork
for them. Instead, draft what the CLIENT (the tenant) should send — e.g. a
written response asserting their rights and objecting to the improper
process, leveraging exactly those same gaps as grounds for the objection.

Where you don't have a specific fact (e.g. an exact date, a party's full
legal name), insert a clear placeholder like [INSERT DATE] rather than
inventing one.

RULE REFERENCE - {title} ({category_id})
Statute: {statute_text}
Required elements: {elements_required}
Procedural notes: {procedural_notes}

CLIENT'S ROLE: {client_role}

ORIGINAL CASE FACTS:
{user_document_text}

COMPLIANCE GAPS FOUND (may be framed as the OTHER party's obligations —
use these as grounds for the client's position, do not fulfill them on
the other party's behalf):
Violations: {violations}
Missing elements: {missing_elements}

Draft a document that serves the client's ({client_role}'s) interests
using these gaps as leverage. Set drafted_for_role to "{client_role}" to
confirm whose side this document is on.

{format_instructions}
"""
)


NOTICE_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal document drafter. The client RECEIVED (or was served
with) the {document_type} described below and needs a response drafted on
their behalf. There is no "compliance gap" to fix here — the client's own
document isn't being checked against anything; you're drafting their
response to something the other side sent or filed.

CRITICAL: the CLIENT's role is "{client_role}". Draft a response that
serves and protects the client's interests as the {client_role} — do not
draft anything that reads as if written by or for the other party.

Use the rule reference to ground the response in the actual applicable
law/procedure (e.g. correct grounds to contest, correct venue, correct
form) — but do not invent facts about the case that aren't in the
{document_type} content below.

Where you don't have a specific fact (e.g. an exact date, a case number,
a party's full legal name), insert a clear placeholder like [INSERT DATE]
rather than inventing one.

RULE REFERENCE - {title} ({category_id})
Statute: {statute_text}
Required elements: {elements_required}
Procedural notes: {procedural_notes}

CLIENT'S ROLE: {client_role}

{document_type} CONTENT RECEIVED BY CLIENT:
{user_document_text}

Draft an appropriate response document for the client. Set
drafted_for_role to "{client_role}" to confirm whose side this document
is on. Since there's no compliance gap here, addresses_gaps should
describe what the response addresses from the {document_type} itself
(e.g. "responds to the allegation in paragraph 2", "contests the claimed
deadline").

{format_instructions}
"""
)


def drafting_node(state: LawFirmState) -> LawFirmState:
    """Two modes, depending on which path led here (see route_after_deadline):

    1. Gap-based drafting: compliance_result exists and compliant=False —
       draft a fix/objection using the specific violations found (existing
       dispute_narrative path behavior).
    2. Notice-response drafting: no compliance_result (court_notice,
       legal_notice, petition, or complaint path) — draft a response to
       what the client received/was served with, using the rule reference
       for grounding but with no "gaps" to work from.

    Both modes use the strong model tier and are perspective-critical:
    read client_role from intake_result so the draft serves the actual
    client, not generically "the document associated with this rule
    category"."""
    rule_doc = state["rule_document"]
    compliance_result = state.get("compliance_result")
    intake_result = state.get("intake_result")

    if rule_doc is None:
        raise ValueError("drafting_node reached without rule_document")

    client_role = intake_result.client_role if intake_result else "the client"

    parser = PydanticOutputParser(pydantic_object=DraftedDocument)
    llm = ChatGroq(model=MODEL_STRONG, temperature=0)

    if compliance_result is not None:
        # Mode 1: gap-based drafting (dispute_narrative path)
        chain = DRAFTING_PROMPT | llm | parser
        result: DraftedDocument = chain.invoke({
            "title": rule_doc.title,
            "category_id": rule_doc.category_id,
            "statute_text": rule_doc.statute_text,
            "elements_required": rule_doc.elements_required,
            "procedural_notes": rule_doc.procedural_notes,
            "client_role": client_role,
            "user_document_text": state["user_document_text"],
            "violations": [v.model_dump() for v in compliance_result.violations],
            "missing_elements": compliance_result.missing_elements,
            "format_instructions": parser.get_format_instructions(),
        })
    else:
        # Mode 2: notice-response drafting (court_notice/legal_notice/
        # petition/complaint path) — no compliance gaps, just a response.
        document_type = intake_result.document_type.value if intake_result else "notice"
        chain = NOTICE_RESPONSE_PROMPT | llm | parser
        result: DraftedDocument = chain.invoke({
            "title": rule_doc.title,
            "category_id": rule_doc.category_id,
            "statute_text": rule_doc.statute_text,
            "elements_required": rule_doc.elements_required,
            "procedural_notes": rule_doc.procedural_notes,
            "client_role": client_role,
            "document_type": document_type,
            "user_document_text": state["user_document_text"],
            "format_instructions": parser.get_format_instructions(),
        })

    return {**state, "drafted_document": result}


# ---------------------------------------------------------------------------
# Node 7: Deadline / Calendar — extracts deadline RULES (not calculated dates)
# ---------------------------------------------------------------------------

DEADLINE_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal deadline extractor. Read the rule reference below and
identify any procedural deadlines or limitation periods relevant to this
case — things like notice periods, response windows, filing deadlines, or
limitation periods to bring a claim.

Do NOT calculate actual calendar dates. You do not reliably know today's
date or the client's exact reference dates. Instead, extract the RULE:
how many days are allowed, and what event starts the countdown (e.g.
"30 days from the date the notice was received"). Leave the actual date
calculation to the client or their lawyer.

If the rule reference doesn't specify any clear day-based deadlines, return
an empty deadlines list rather than inventing one.

RULE REFERENCE - {title} ({category_id})
Procedural notes: {procedural_notes}
Elements required: {elements_required}

CASE CONTEXT (for relevance only, not for date calculation):
{user_document_text}

{format_instructions}
"""
)


def deadline_node(state: LawFirmState) -> LawFirmState:
    """Cheap/fast model tier — this is extracting deadline RULES from
    already-retrieved rule text, not new legal reasoning or date math
    (which LLMs are unreliable at — see DEADLINE_PROMPT). Runs on BOTH
    the compliance path and the redline path, before the final split to
    drafting/summarization."""
    rule_doc = state["rule_document"]
    if rule_doc is None:
        raise ValueError("deadline_node reached with no rule_document — check graph routing")

    parser = PydanticOutputParser(pydantic_object=DeadlineResult)
    llm = ChatGroq(model=MODEL_CHEAP, temperature=0)
    chain = DEADLINE_PROMPT | llm | parser

    result: DeadlineResult = chain.invoke({
        "title": rule_doc.title,
        "category_id": rule_doc.category_id,
        "procedural_notes": rule_doc.procedural_notes,
        "elements_required": rule_doc.elements_required,
        "user_document_text": state["user_document_text"],
        "format_instructions": parser.get_format_instructions(),
    })

    return {**state, "deadline_result": result}


def route_after_deadline(state: LawFirmState) -> str:
    """Decides whether drafting_node runs, now that deadlines have been
    surfaced regardless of path:

    - dispute_narrative, not compliant -> drafting (gap-based, Mode 1)
    - court_notice, legal_notice, petition, complaint -> drafting
      (notice-response, Mode 2 - the client needs a response drafted to
      what they received/were served with)
    - dispute_narrative, compliant -> summarization (nothing to draft)
    - judgment -> summarization (informational; no response drafted by
      default - a judgment is a decided outcome, not something to respond
      to the same way a notice or petition is)
    - contract_for_review -> summarization (no drafting step exists for
      contract redlines yet - see README 'Next steps')"""
    compliance_result = state.get("compliance_result")
    if compliance_result is not None and not compliance_result.compliant:
        return "drafting"

    intake_result = state.get("intake_result")
    document_type = intake_result.document_type if intake_result else None

    if document_type in (
        DocumentType.COURT_NOTICE,
        DocumentType.LEGAL_NOTICE,
        DocumentType.PETITION,
        DocumentType.COMPLAINT,
    ):
        return "drafting"

    return "summarization"


# ---------------------------------------------------------------------------
# Node 6: Summarization — client-facing plain-language summary
# ---------------------------------------------------------------------------

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_template(
    """You are explaining a legal compliance result to a client in plain
language. The client is a "{client_role}" with no legal training — do NOT
use legal jargon, statute numbers, or technical terms. Write like you're
explaining this to a friend, clearly and warmly but factually. Do not
soften or hide bad news, and do not add reassurance that isn't warranted.

CASE SUMMARY: {issue_description}

COMPLIANCE FINDING:
Compliant: {compliant}
Issues found: {violations_and_missing}
Technical summary: {technical_summary}
{drafted_document_section}
{deadline_section}
Write a plain_summary (3-5 sentences), a key_takeaway (one sentence, the
single most important thing to know), and next_steps (concrete, actionable,
in plain language — e.g. "Send the attached letter to your landlord by
certified mail" not "pursue your remedies under the Act").

{format_instructions}
"""
)


def summarization_node(state: LawFirmState) -> LawFirmState:
    """Cheap/fast model tier — this is tone and format transformation of
    already-completed analysis, not new legal reasoning, so it doesn't need
    the strong tier. Handles THREE possible upstream states: compliance_result
    (dispute_narrative path, possibly with a drafted_document), redline_result
    (contract_for_review path), or neither (court_notice/legal_notice/
    petition/complaint/judgment path, which skips compare/contract_redline
    entirely since there's nothing to compliance-check on a received/filed
    document). Also incorporates deadline_result if any deadlines were
    found."""
    intake_result = state.get("intake_result")
    compliance_result = state.get("compliance_result")
    redline_result = state.get("redline_result")
    drafted_document = state.get("drafted_document")
    deadline_result = state.get("deadline_result")

    client_role = intake_result.client_role if intake_result else "the client"
    issue_description = intake_result.issue_description if intake_result else state.get("user_document_text", "")

    if redline_result is not None:
        violations_and_missing = (
            [f"{i.risk_level.value}: {i.issue}" for i in redline_result.issues]
            + redline_result.missing_protections
        )
        compliant = redline_result.overall_risk.value == "low"
        technical_summary = redline_result.summary
    elif compliance_result is not None:
        violations_and_missing = (
            [f"{v.severity.value}: {v.issue}" for v in compliance_result.violations]
            + compliance_result.missing_elements
        )
        compliant = compliance_result.compliant
        technical_summary = compliance_result.summary
    else:
        # court_notice / legal_notice / petition / complaint / judgment path:
        # no compliance check ever ran (compare/contract_redline were
        # skipped by design), so there's no violation/risk finding to
        # report — just summarize the document itself and any drafted
        # response or deadlines.
        violations_and_missing = []
        compliant = "not applicable — this document type is not compliance-checked"
        technical_summary = (
            intake_result.issue_description if intake_result
            else "No compliance or redline analysis applies to this document type."
        )

    drafted_document_section = (
        f"\nA draft document was prepared to help address this: \"{drafted_document.title}\" "
        f"({drafted_document.document_type}). Mention in next_steps that this draft is available "
        f"for the client to review and use.\n"
        if drafted_document else "\n"
    )

    deadline_section = (
        f"\nRelevant deadlines were found: "
        f"{[(d.deadline_type, d.days_from_reference, d.reference_event) for d in deadline_result.deadlines]}. "
        f"Mention these in next_steps if they are time-sensitive.\n"
        if deadline_result and deadline_result.deadlines else "\n"
    )

    parser = PydanticOutputParser(pydantic_object=ClientSummary)
    llm = ChatGroq(model=MODEL_CHEAP, temperature=0)
    chain = SUMMARIZATION_PROMPT | llm | parser

    result: ClientSummary = chain.invoke({
        "client_role": client_role,
        "issue_description": issue_description,
        "compliant": compliant,
        "violations_and_missing": violations_and_missing,
        "technical_summary": technical_summary,
        "drafted_document_section": drafted_document_section,
        "deadline_section": deadline_section,
        "format_instructions": parser.get_format_instructions(),
    })

    return {**state, "client_summary": result}


# ---------------------------------------------------------------------------
# Node 8: Notification — email the client when their case has deadlines
# ---------------------------------------------------------------------------

def notify_node(state: LawFirmState) -> LawFirmState:
    """Sends the client an email when their case has a work-related
    deadline attached. Runs AFTER summarization so the email can use the
    plain-language client_summary (key_takeaway, next_steps) instead of
    the raw deadline rule text — much more useful to a non-lawyer.

    No-op (not an error) if client_email wasn't provided when the graph
    was invoked, or if no deadlines were found — most runs won't need an
    email, and that's expected, not a failure."""
    client_email = state.get("client_email")
    deadline_result = state.get("deadline_result")
    client_summary = state.get("client_summary")

    if not client_email or not deadline_result or not deadline_result.deadlines:
        return {**state, "notification_sent": False}

    intake_result = state.get("intake_result")
    issue = intake_result.issue_description if intake_result else "your case"

    deadline_lines = "\n".join(
        f"- {d.deadline_type}: {d.description} "
        + (f"({d.days_from_reference} days from {d.reference_event})"
           if d.days_from_reference is not None
           else f"(from {d.reference_event})")
        for d in deadline_result.deadlines
    )

    # Explicit dates (e.g. "31 July 2026") are pulled directly from the
    # client's own text with dateparser — NOT by the LLM, which is
    # deliberately kept out of date arithmetic (see DEADLINE_PROMPT).
    # This is a best-effort scan and can misfire on incidental dates
    # (e.g. a past payment date mentioned in passing), so it's labeled
    # clearly and never presented as a verified deadline.
    explicit_date_line = ""
    raw_text = state.get("user_document_text") or state.get("raw_user_message") or ""
    found = search_dates(raw_text, settings={"PREFER_DATES_FROM": "future"})
    if found:
        dates_str = ", ".join(sorted({d[0] for d in found}))
        explicit_date_line = (
            f"\nDate(s) mentioned in your document: {dates_str}\n"
            "(Auto-detected from your text — please verify this is the "
            "correct date before relying on it.)\n"
        )

    body_parts = [f"Hello,\n\nThis is an update regarding: {issue}\n"]
    if client_summary:
        body_parts.append(f"Summary: {client_summary.plain_summary}\n")
        body_parts.append(f"Key takeaway: {client_summary.key_takeaway}\n")
    if explicit_date_line:
        body_parts.append(explicit_date_line)
    body_parts.append(f"Deadlines to be aware of:\n{deadline_lines}\n")
    body_parts.append(f"\n{deadline_result.disclaimer}")

    sent = send_email(
        to_email=client_email,
        subject="Action needed: deadline update on your case",
        body="\n".join(body_parts),
    )

    return {**state, "notification_sent": sent}


# ---------------------------------------------------------------------------
# Human-in-the-loop gate (placeholder — wire to LangGraph's interrupt() )
# ---------------------------------------------------------------------------

def human_review_gate(state: LawFirmState) -> LawFirmState:
    """Placeholder node. In a real deployment, use LangGraph's `interrupt`
    mechanism here to pause execution until a human approves/edits the
    output, rather than auto-approving as this stub does. This is
    especially important now that drafting_node may produce a document
    meant for actual use — a human MUST review before it goes out."""
    return {**state, "human_approved": True}