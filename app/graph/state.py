"""
Shared state that flows through the LangGraph pipeline.

Keep this lean — each node should only read/write the fields it actually
needs (see the earlier "state bloat" discussion). Avoid dumping every
upstream agent's full output into every downstream node's context.
"""
from typing import TypedDict, Optional
from app.schemas import (
    ClassifyResponse, RuleDocument, ComplianceResult,
    VerificationResult, IntakeResult, DraftedDocument, ClientSummary,
    RedlineResult, DeadlineResult,
)


class LawFirmState(TypedDict, total=False):
    # Input — raw, possibly messy user message (new entry point)
    raw_user_message: str

    # Node 0: Intake
    intake_result: Optional[IntakeResult]
    needs_clarification: bool
    clarification_question: Optional[str]

    # Feeds into classification — either from intake's synthesis, or
    # provided directly if you bypass intake (e.g. running graph.py's
    # standalone test with a pre-written sample document)
    user_document_text: str

    # Node 1: Classify
    classification: Optional[ClassifyResponse]

    # Node 2: Retrieve (direct fetch, no search)
    rule_document: Optional[RuleDocument]
    no_confident_match: bool

    # Node 3: Compare / Compliance check (dispute_narrative path only)
    compliance_result: Optional[ComplianceResult]

    # Node 3b: Contract Review / Redline (contract_for_review path only —
    # mutually exclusive with compliance_result; routing picks one path)
    redline_result: Optional[RedlineResult]

    # Node 4: Citation Verification
    verification_result: Optional[VerificationResult]
    # Set by citation_verification_node when sending compare_node back for
    # a retry — compare_node reads this to correct itself, cleared once used.
    verification_feedback: Optional[str]

    # Node 5: Document Drafting (only runs if compliance_result.compliant is False)
    drafted_document: Optional[DraftedDocument]

    # Node 7: Deadline / Calendar — runs on both paths before summarization
    deadline_result: Optional[DeadlineResult]

    # Node 6: Summarization — client-facing, runs regardless of compliant/not
    client_summary: Optional[ClientSummary]

    # Node 8: Notification — email address to notify about deadlines.
    # Not collected by intake_node (it's contact info, not case info) —
    # pass it in directly alongside raw_user_message when invoking the
    # graph / calling the API.
    client_email: Optional[str]
    notification_sent: bool

    # Human-in-the-loop gate
    requires_human_review: bool
    human_approved: Optional[bool]

    # Iteration guard (prevents infinite loops per the rate-limit design)
    retry_count: int