r"""
Wires up the LangGraph pipeline:

  intake --> [ready?] --> classify_and_retrieve --> [document_type route] --> ...
         \-> [needs clarification] --> END

  route_after_classification branches on document_type (7 possible values,
  see schemas.DocumentType) - ONLY the agents relevant to that type run:
    - no confident category match -> human_review
    - contract_for_review -> contract_redline -> deadline -> [route] -> ...
    - dispute_narrative   -> compare -> citation_verification -> [route] -> ...
    - court_notice, legal_notice, judgment, petition, complaint ->
      straight to deadline, SKIPPING compare/citation_verification/redline
      entirely (these are received/filed documents, not something to
      compliance-check)

  route_after_verification (compare path only):
    - not verified, retries left -> back to compare
    - verified (or retries exhausted) -> deadline

  route_after_deadline (all paths converge here):
    - dispute_narrative, NOT compliant -> drafting (gap-based fix/objection)
    - court_notice, legal_notice, petition, complaint -> drafting
      (notice-response - client needs a reply drafted)
    - judgment -> summarization (informational, no default response drafted)
    - dispute_narrative compliant, or contract_for_review -> summarization

Intake decides document_type up front (see DocumentType enum in schemas.py)
- this determines which agents actually run, so a court notice never
  triggers a compliance check, and a judgment never triggers drafting
  by default.

Citation verification can send the case back to compare_node up to
MAX_VERIFICATION_RETRIES times (see nodes.py). Note: the contract_redline
path does NOT go through citation_verification yet, and drafting_node has
two internal modes (gap-based vs notice-response) depending on which path
led to it - see README.

Deadline extraction runs on every path (except when routed straight to
human_review for a low-confidence category match) before the final split,
so procedural deadlines are surfaced regardless of document type.
Summarization runs last on every path as the final client-facing
translation step.
"""
from langgraph.graph import StateGraph, END

from app.graph.state import LawFirmState
from app.graph.nodes import (
    intake_node,
    route_after_intake,
    classify_and_retrieve_node,
    route_after_classification,
    compare_node,
    citation_verification_node,
    route_after_verification,
    drafting_node,
    contract_redline_node,
    deadline_node,
    route_after_deadline,
    summarization_node,
    notify_node,
    human_review_gate,
)


def build_graph():
    graph = StateGraph(LawFirmState)

    graph.add_node("intake", intake_node)
    graph.add_node("classify_and_retrieve", classify_and_retrieve_node)
    graph.add_node("compare", compare_node)
    graph.add_node("citation_verification", citation_verification_node)
    graph.add_node("drafting", drafting_node)
    graph.add_node("contract_redline", contract_redline_node)
    graph.add_node("deadline", deadline_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("notify", notify_node)
    graph.add_node("human_review", human_review_gate)

    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "classify_and_retrieve": "classify_and_retrieve",
            "end_for_clarification": END,
        },
    )

    graph.add_conditional_edges(
        "classify_and_retrieve",
        route_after_classification,
        {
            "compare": "compare",
            "contract_redline": "contract_redline",
            "deadline": "deadline",
            "human_review": "human_review",
        },
    )

    graph.add_edge("compare", "citation_verification")

    graph.add_conditional_edges(
        "citation_verification",
        route_after_verification,
        {
            "compare": "compare",
            "deadline": "deadline",
        },
    )

    graph.add_edge("contract_redline", "deadline")

    graph.add_conditional_edges(
        "deadline",
        route_after_deadline,
        {
            "drafting": "drafting",
            "summarization": "summarization",
        },
    )

    graph.add_edge("drafting", "summarization")
    graph.add_edge("summarization", "notify")
    graph.add_edge("notify", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile()


def print_result(result: dict):
    if result.get("needs_clarification"):
        print("--- Needs Clarification ---")
        print(result.get("clarification_question"))
        return

    print("--- Intake Result ---")
    if result.get("intake_result"):
        print(result["intake_result"].model_dump_json(indent=2))
    print("--- Rule Document ---")
    if result.get("rule_document"):
        print(result["rule_document"].category_id)
    print("--- Compliance Result ---")
    if result.get("compliance_result"):
        print(result["compliance_result"].model_dump_json(indent=2))
    else:
        print("(none — redline path)")
    print("--- Redline Result ---")
    if result.get("redline_result"):
        print(result["redline_result"].model_dump_json(indent=2))
    else:
        print("(none — compliance path)")
    print("--- Deadline Result ---")
    if result.get("deadline_result"):
        print(result["deadline_result"].model_dump_json(indent=2))
    print("--- Drafted Document ---")
    if result.get("drafted_document"):
        print(result["drafted_document"].model_dump_json(indent=2))
    else:
        print("(none)")
    print("--- Client Summary ---")
    if result.get("client_summary"):
        print(result["client_summary"].model_dump_json(indent=2))
    print("--- Retry Count ---")
    print(result.get("retry_count", 0))
    print("--- Notification Sent ---")
    print(result.get("notification_sent", False))


if __name__ == "__main__":
    app_graph = build_graph()

    print("=== Test 1: dispute narrative ===\n")
    narrative_message = """
    my landlord Mr. Riaz wants me to leave the shop I've been renting for
    3 years in Lahore. we never had a written agreement, just a verbal
    deal, Rs 25000 a month, paid till March 2026. he told me verbally to
    leave within 7 days because his son needs the shop. I refused since
    I've been a good tenant. now he changed the lock while I was away and
    I can't get my stuff out.
    """
    result1 = app_graph.invoke({"raw_user_message": narrative_message, "retry_count": 0})
    print_result(result1)

    print("\n\n=== Test 2: contract for review ===\n")
    contract_message = """
    Please review this shop lease clause for risk: "The Tenant shall pay
    rent of Rs 25,000 per month. The Landlord may terminate this agreement
    at any time without notice and without cause. The Tenant waives any
    right to dispute termination. Security deposit of Rs 50,000 is
    non-refundable under any circumstances." I am the tenant being asked
    to sign this in Lahore.
    """
    result2 = app_graph.invoke({"raw_user_message": contract_message, "retry_count": 0})
    print_result(result2)

    print("\n\n=== Test 3: legal notice received (new path - should SKIP compare/citation_verification) ===\n")
    notice_message = """
    Here is a notice I received: "NOTICE TO VACATE. To the Tenant occupying
    the shop at [address], Lahore. You are hereby directed to vacate the
    premises within 7 days of this notice, failing which legal action will
    be taken. Signed, Landlord Mr. Riaz." I am the tenant who received this
    and want to know how to respond.
    """
    result3 = app_graph.invoke({"raw_user_message": notice_message, "retry_count": 0})
    print_result(result3)
    print("\n--- Sanity check: compliance_result should be None (compare_node skipped) ---")
    print("compliance_result is None:", result3.get("compliance_result") is None)
    print("intake document_type:", result3["intake_result"].document_type if result3.get("intake_result") else None)