from typing import TypedDict, Optional, List


class SupportState(TypedDict, total=False):
    # Module 1 - Conversation State & Context Management
    user_message: str
    history: List[str]
    trace: List[str]
    approval_replay: bool

    # Module 2 - Intent Classification & Sentiment Triage
    intent: str
    sentiment: str
    urgency: str

    # Module 3 - Specialist Router
    specialist: str

    # Module 4 - Tool-Calling & Human-in-the-Loop Execution
    booking_id: Optional[str]
    tool_result: Optional[dict]
    needs_approval: bool
    user_approved: Optional[bool]

    # Module 5 - Response Generation & Policy Grounding
    policy_context: Optional[str]
    response: str

    # Module 6 - Escalation & Feedback Logging
    escalated: bool
    resolved: bool
