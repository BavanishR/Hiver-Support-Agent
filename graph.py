from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import (
    classify_intent_node,
    clarify_node,
    escalate_node,
    generate_reply_node
)

ALWAYS_ESCALATE_NO_REPLY = {"ACCOUNT_ACCESS"}

ESCALATE_AFTER_REPLY = {"REFUND_AND_RETURN", "WRONG_DAMAGED_ITEM"}

def route_after_classification(state: AgentState) -> str:
    intent = state.get("intent")
    confidence = state.get("confidence_score", 0.0)
    clarify_count = state.get("clarify_count", 0)

    if intent in ALWAYS_ESCALATE_NO_REPLY:
        return "escalate"

    if intent == "UNCLEAR_VAGUE" or confidence < 0.70:
        if clarify_count >= 2:
            return "escalate"
        return "clarify"

    return "generate_reply"


def route_after_reply(state: AgentState) -> str:
    intent = state.get("intent")
    if intent in ESCALATE_AFTER_REPLY:
        return "escalate"
    return "end"


workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("escalate", escalate_node)
workflow.add_node("generate_reply", generate_reply_node)


workflow.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {"clarify": "clarify", "escalate": "escalate", "generate_reply": "generate_reply"}
)

workflow.add_conditional_edges(
    "generate_reply",
    route_after_reply,
    {
        "escalate": "escalate",
        "end": END
    }
)

workflow.add_edge(START, "classify_intent")
workflow.add_edge("clarify", END)
workflow.add_edge("escalate", END)

app = workflow.compile()