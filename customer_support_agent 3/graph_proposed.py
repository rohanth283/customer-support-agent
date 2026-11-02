"""
Proposed Customer Support Agent - LangGraph implementation.

Implements the six functional modules defined in the Proposed Methodology:
  1. Conversation State & Context Management
  2. Intent Classification & Sentiment Triage
  3. Specialist Router (Assistant Graph)
  4. Tool-Calling & Human-in-the-Loop Execution   (ReAct: think -> act -> observe)
  5. Response Generation & Policy Grounding
  6. Escalation & Feedback Logging
"""

import re
from langgraph.graph import StateGraph, END

from state import SupportState
from llm import get_llm
from tools import backend

llm = get_llm()

BOOKING_ID_RE = re.compile(
    r"\b([A-Z]{3}-[A-Z]{3}-\d{2}|HTL-\d{4}|CAR-\d{3})\b",
    re.IGNORECASE,
)
_LOOKUP_RE = re.compile(r"\b(status|details|detail|lookup|check|track)\b|\blook\s+up\b", re.IGNORECASE)
_NEW_BOOKING_RE = re.compile(r"\b(book|reserve|need|looking|rent|rental|pickup)\b", re.IGNORECASE)

SENSITIVE_INTENTS = {"cancel_flight", "cancel_hotel", "cancel_car", "cancel_generic"}


# ---------------------------------------------------------------- Module 1
def update_context(state: SupportState) -> SupportState:
    history = state.get("history", [])
    if not state.get("approval_replay"):
        history = history + [state["user_message"]]
    return {
        "history": history,
        "approval_replay": False,
        "trace": state.get("trace", []) + ["Context & conversation state"],
    }


# ---------------------------------------------------------------- Module 2
def triage(state: SupportState) -> SupportState:
    text = state["user_message"]
    intent = llm.classify_intent(text)
    sentiment, urgency = llm.classify_sentiment(text)
    match = BOOKING_ID_RE.search(text)
    booking_id = match.group(0).upper() if match else None
    return {
        "intent": intent,
        "sentiment": sentiment,
        "urgency": urgency,
        "booking_id": booking_id,
        "trace": state.get("trace", []) + ["Intent & sentiment triage"],
    }


# ---------------------------------------------------------------- Module 3
def _specialist_for_booking(booking_id: str) -> str:
    if booking_id.startswith("HTL"):
        return "hotel_specialist"
    if booking_id.startswith("CAR"):
        return "car_specialist"
    return "flight_specialist"


def route(state: SupportState) -> str:
    intent = state["intent"]
    if intent in ("flight_booking", "cancel_flight"):
        return "flight_specialist"
    if intent in ("hotel_booking", "cancel_hotel"):
        return "hotel_specialist"
    if intent in ("car_rental", "cancel_car"):
        return "car_specialist"
    if intent == "policy_query":
        return "policy_specialist"
    if intent == "cancel_generic" and state.get("booking_id"):
        return _specialist_for_booking(state["booking_id"])
    return "general_specialist"


def _is_new_booking(text: str) -> bool:
    if _LOOKUP_RE.search(text):
        return False
    return _NEW_BOOKING_RE.search(text) is not None


# ---------------------------------------------------------- Module 4 nodes
def flight_specialist(state: SupportState) -> SupportState:
    return _domain_specialist(state, "flight", backend.search_flight, backend.cancel_flight, backend.book_flight)


def hotel_specialist(state: SupportState) -> SupportState:
    return _domain_specialist(state, "hotel", backend.search_hotel, backend.cancel_hotel, backend.book_hotel)


def car_specialist(state: SupportState) -> SupportState:
    return _domain_specialist(state, "car", backend.search_car, backend.cancel_car, backend.book_car)


def _domain_specialist(state: SupportState, domain: str, search_fn, cancel_fn, book_fn) -> SupportState:
    booking_id = state.get("booking_id")
    intent = state["intent"]
    is_cancel = intent.startswith("cancel")
    specialist = f"{domain}_specialist"

    if not booking_id:
        if is_cancel:
            return {
                "tool_result": {"ok": False, "message": f"No {domain} booking id found in message."},
                "needs_approval": False,
                "specialist": specialist,
                "trace": state.get("trace", []) + [f"{domain.title()} specialist & tools"],
            }
        if _is_new_booking(state.get("user_message", "")):
            return {
                "tool_result": book_fn(state.get("user_message", "")),
                "needs_approval": False,
                "specialist": specialist,
                "trace": state.get("trace", []) + [f"{domain.title()} specialist & tools"],
            }
        return {
            "tool_result": {"ok": False, "message": f"No {domain} booking id found in message."},
            "needs_approval": False,
            "specialist": specialist,
            "trace": state.get("trace", []) + [f"{domain.title()} specialist & tools"],
        }

    if is_cancel:
        # ReAct-style grounding checkpoint: sensitive action -> human approval required
        if state.get("user_approved") is not True:
            return {
                "needs_approval": True,
                "specialist": specialist,
                "tool_result": {"ok": None, "message": f"Confirmation required to cancel {domain} booking {booking_id}."},
                "trace": state.get("trace", []) + [f"{domain.title()} specialist & approval gate"],
            }
        result = cancel_fn(booking_id)
        return {"tool_result": result, "needs_approval": False, "specialist": specialist,
                "trace": state.get("trace", []) + [f"{domain.title()} specialist & tools"]}

    record = search_fn(booking_id)
    if record is None:
        result = {"ok": False, "message": f"No {domain} booking found with id {booking_id}."}
    else:
        result = {"ok": True, "record": record}
    return {"tool_result": result, "needs_approval": False, "specialist": specialist,
            "trace": state.get("trace", []) + [f"{domain.title()} specialist & tools"]}


def policy_specialist(state: SupportState) -> SupportState:
    passage = backend.search_policy(state["user_message"])
    return {"policy_context": passage, "tool_result": {"ok": True, "message": "policy retrieved"},
            "needs_approval": False, "specialist": "policy_specialist",
            "trace": state.get("trace", []) + ["Policy retrieval"]}


def general_specialist(state: SupportState) -> SupportState:
    return {
        "tool_result": {
            "ok": False,
            "message": (
                "I can help with flight, hotel, and car bookings, cancellations, and policies. "
                "Share a booking id or a policy question, or I can connect you with an agent."
            ),
        },
        "needs_approval": False,
        "specialist": "general_specialist",
        "trace": state.get("trace", []) + ["General support specialist"],
    }


# ---------------------------------------------------------------- Module 5
def generate_response(state: SupportState) -> SupportState:
    intent = state["intent"]
    result = state.get("tool_result") or {}

    if state.get("needs_approval"):
        booking_id = state.get("booking_id") or "the selected booking"
        domain = (state.get("specialist") or "").removesuffix("_specialist")
        action = f"{domain.title()} booking {booking_id}" if domain else f"booking {booking_id}"
        response = f"Please confirm cancellation of {action}. I will not cancel it until you approve."
        resolved = False
    elif intent == "policy_query":
        response = f"Here is our policy: {state.get('policy_context')}"
        resolved = True
    elif result.get("ok") is True and "record" in result:
        response = f"Found your {state['specialist'].split('_')[0]} booking: {result['record']}."
        resolved = True
    elif result.get("ok") is True:
        response = result.get("message", "Done.")
        resolved = True
    else:
        response = result.get("message", "I'm sorry, I could not complete that request.")
        resolved = False

    return {"response": llm.generate_response(response), "resolved": resolved,
            "trace": state.get("trace", []) + ["Response generation & policy grounding"]}


# ---------------------------------------------------------------- Module 6
def escalate_or_log(state: SupportState) -> SupportState:
    escalate = (
        state.get("sentiment") == "negative" and state.get("urgency") == "high"
    ) or (not state.get("resolved", False) and not state.get("needs_approval", False))
    return {"escalated": bool(escalate),
            "trace": state.get("trace", []) + ["Escalation & outcome logging"]}


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("update_context", update_context)
    g.add_node("triage", triage)
    g.add_node("flight_specialist", flight_specialist)
    g.add_node("hotel_specialist", hotel_specialist)
    g.add_node("car_specialist", car_specialist)
    g.add_node("policy_specialist", policy_specialist)
    g.add_node("general_specialist", general_specialist)
    g.add_node("generate_response", generate_response)
    g.add_node("escalate_or_log", escalate_or_log)

    g.set_entry_point("update_context")
    g.add_edge("update_context", "triage")
    g.add_conditional_edges("triage", route, {
        "flight_specialist": "flight_specialist",
        "hotel_specialist": "hotel_specialist",
        "car_specialist": "car_specialist",
        "policy_specialist": "policy_specialist",
        "general_specialist": "general_specialist",
    })
    for node in ("flight_specialist", "hotel_specialist", "car_specialist",
                 "policy_specialist", "general_specialist"):
        g.add_edge(node, "generate_response")
    g.add_edge("generate_response", "escalate_or_log")
    g.add_edge("escalate_or_log", END)

    return g.compile()


PROPOSED_GRAPH = build_graph()


if __name__ == "__main__":
    backend.reset()
    result = PROPOSED_GRAPH.invoke({"user_message": "Cancel my flight BLR-DEL-01"})
    print(result["response"])
    result["user_approved"] = True
    result = PROPOSED_GRAPH.invoke(result)
    print(result["response"])
