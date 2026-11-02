"""
Pluggable "LLM" interface used throughout the agent graph.

Two backends are provided:

1. RuleBasedLLM (default) - a deterministic, keyword-driven stand-in for a
   GPT-4-class model. It exists so that the full LangGraph pipeline can be
   built, run, tested, and evaluated end-to-end WITHOUT an OpenAI API key or
   network access - which matters for reproducible grading/demo environments.
   It is intentionally simple: it is not a claim of NLU accuracy, only a way
   to exercise every node, edge, and interrupt in the graph with real,
   inspectable inputs/outputs.

2. OpenAILLM - a thin wrapper around langchain_openai.ChatOpenAI, used
   automatically if OPENAI_API_KEY is set in the environment. This is the
   production backend referenced in the Proposed Methodology (GPT-4-class
   API / open-weight alternative).

Swapping backends requires no changes anywhere else in the codebase, since
every node calls `get_llm()` and only depends on the .classify_intent(),
.classify_sentiment(), and .generate_response() interface below.
"""

import os
import re
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def _load_local_env_file() -> None:
    """Load simple KEY=VALUE entries from this project’s ignored .env file."""
    env_path = Path(__file__).with_name(".env")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].lstrip()
        key, separator, value = entry.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_local_env_file()


KNOWN_INTENTS = {
    "flight_booking", "hotel_booking", "car_rental", "policy_query",
    "cancel_flight", "cancel_hotel", "cancel_car", "cancel_generic", "general_query",
}

_INTENT_FOR_DOMAIN = {
    "flight": "flight_booking",
    "hotel": "hotel_booking",
    "car": "car_rental",
}


def _extract_intent(response: str) -> str:
    text = response.strip().lower()
    for label in sorted(KNOWN_INTENTS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", text):
            return label
    return "general_query"


def _extract_sentiment_urgency(response: str):
    text = response.strip().lower()
    sentiment_match = re.search(r"\b(positive|neutral|negative)\b", text)
    urgency_match = re.search(r"\b(low|medium|high)\b", text)
    sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
    urgency = urgency_match.group(1) if urgency_match else "low"
    return sentiment, urgency


def _preserve_grounded_facts(source: str, response: str) -> str:
    """Use the deterministic source text if a model drops IDs or numeric facts."""
    booking_ids = re.findall(r"\b(?:[A-Z]{3}-[A-Z]{3}-\d{2}|HTL-\d{4}|CAR-\d{3}|FLT-\d{4})\b", source, re.I)
    numeric_facts = re.findall(r"\b\d+(?:-\d+)?(?:%|kg)?\b", source, re.I)
    response_lower = response.lower()
    if any(value.lower() not in response_lower for value in booking_ids + numeric_facts):
        return source
    return response


class BaseLLM:
    def classify_intent(self, text: str) -> str:
        raise NotImplementedError

    def classify_sentiment(self, text: str):
        raise NotImplementedError

    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError


class RuleBasedLLM(BaseLLM):
    """Deterministic offline stand-in (see module docstring)."""

    provider_name = "Rule-based (offline)"

    POLICY_KEYWORDS = ["policy", "refund", "refunds", "baggage", "allowance",
                       "reschedule", "rescheduling", "rule", "rules", "fee", "fees"]
    FLIGHT_KEYWORDS = ["flight", "flights", "fly", "airport", "plane", "boarding"]
    HOTEL_KEYWORDS = ["hotel", "hotels", "room", "rooms", "checkin", "check-in", "stay"]
    CAR_KEYWORDS = ["car", "cars", "rental", "rentals", "vehicle", "pickup", "drive"]

    NEGATIVE_WORDS = ["angry", "furious", "terrible", "worst", "unacceptable",
                       "frustrated", "disappointed", "horrible", "scam", "never"]
    URGENT_WORDS = ["urgent", "immediately", "asap", "now", "emergency"]

    # A request to perform a cancellation, not a mention of the word inside a policy question.
    _IMPERATIVE_CANCEL = re.compile(
        r"(?:^|[.!?]\s*)cancel\b"
        r"|\bplease\s+cancel\b"
        r"|\b(?:want|need|like)\s+to\s+cancel\b"
        r"|\bcancel\s+(?:my|the|this|booking|it)\b",
        re.IGNORECASE,
    )

    def _hits(self, text: str, keywords) -> int:
        return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text))

    def _domain_from_id(self, text: str):
        if re.search(r"\bHTL-\d{4}\b", text, re.IGNORECASE):
            return "hotel"
        if re.search(r"\bCAR-\d{3}\b", text, re.IGNORECASE):
            return "car"
        if re.search(r"\b[A-Za-z]{3}-[A-Za-z]{3}-\d{2}\b", text):
            return "flight"
        return None

    def _domain_from_keywords(self, text: str):
        scores = {
            "flight": self._hits(text, self.FLIGHT_KEYWORDS),
            "hotel": self._hits(text, self.HOTEL_KEYWORDS),
            "car": self._hits(text, self.CAR_KEYWORDS),
        }
        best_score = max(scores.values())
        if best_score == 0:
            return None
        winners = [domain for domain, score in scores.items() if score == best_score]
        if len(winners) == 1:
            return winners[0]
        id_domain = self._domain_from_id(text)
        if id_domain in winners:
            return id_domain
        return winners[0]

    def classify_intent(self, text: str) -> str:
        text_l = text.lower()
        policy_score = self._hits(text_l, self.POLICY_KEYWORDS)
        domain = self._domain_from_keywords(text_l) or self._domain_from_id(text)
        imperative_cancel = self._IMPERATIVE_CANCEL.search(text_l) is not None

        # Policy questions ("what is the cancellation fee / refund") are not cancel actions,
        # even when they mention a domain or a booking id.
        if policy_score > 0 and not imperative_cancel:
            return "policy_query"
        if imperative_cancel:
            return f"cancel_{domain}" if domain else "cancel_generic"
        if domain:
            return _INTENT_FOR_DOMAIN[domain]
        return "general_query"

    def classify_sentiment(self, text: str):
        text_l = text.lower()
        negative = any(re.search(rf"\b{re.escape(w)}\b", text_l) for w in self.NEGATIVE_WORDS)
        urgent = any(re.search(rf"\b{re.escape(w)}\b", text_l) for w in self.URGENT_WORDS) or "!" in text
        sentiment = "negative" if negative else "neutral"
        urgency = "high" if (negative and urgent) else ("medium" if (negative or urgent) else "low")
        return sentiment, urgency

    def generate_response(self, prompt: str) -> str:
        # Extremely small templated generator: echoes the structured facts
        # it was given rather than "hallucinating" free text. Real system
        # would call an LLM here with the same prompt.
        return prompt


class OpenAILLM(BaseLLM):
    """Production backend - requires OPENAI_API_KEY and langchain-openai."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI
        self.client = ChatOpenAI(model=model, temperature=0)
        self.provider_name = f"OpenAI ({model})"

    def classify_intent(self, text: str) -> str:
        resp = self.client.invoke(
            f"Classify the customer support intent of this message into one of: "
            f"flight_booking, hotel_booking, car_rental, policy_query, "
            f"cancel_flight, cancel_hotel, cancel_car, cancel_generic, general_query. "
            f"Reply with only the label.\n\nMessage: {text}"
        )
        return _extract_intent(resp.content)

    def classify_sentiment(self, text: str):
        resp = self.client.invoke(
            f"Classify sentiment (positive/neutral/negative) and urgency "
            f"(low/medium/high) of this message as 'sentiment,urgency'.\n\n{text}"
        )
        return _extract_sentiment_urgency(resp.content)

    def generate_response(self, prompt: str) -> str:
        resp = self.client.invoke(prompt)
        return _preserve_grounded_facts(prompt, resp.content.strip())


class OllamaLLM(BaseLLM):
    """Local Ollama adapter using its native, non-streaming chat endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
        self.provider_name = f"Ollama ({self.model})"

    def _generate(self, instruction: str, prompt: str, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start the Ollama app/service and try again."
            ) from exc
        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned no chat message. Check that the model is installed.") from exc

    def classify_intent(self, text: str) -> str:
        labels = ", ".join(sorted(KNOWN_INTENTS))
        response = self._generate(
            f"Classify this customer-support request. Reply with exactly one label from: {labels}. "
            "Questions about policy, fees, refunds, baggage, or rescheduling are policy_query, "
            "not requests to execute a cancellation.",
            text,
            max_tokens=32,
        )
        return _extract_intent(response)

    def classify_sentiment(self, text: str):
        response = self._generate(
            "Classify sentiment as positive, neutral, or negative and urgency as low, medium, or high. "
            "Reply only as sentiment,urgency (for example neutral,low).",
            text,
            max_tokens=16,
        )
        return _extract_sentiment_urgency(response)

    def generate_response(self, prompt: str) -> str:
        response = self._generate(
            "You are a concise customer-support agent. Use only the provided tool result or policy passage. "
            "Do not claim an action was completed unless the provided result confirms it. Preserve every booking ID, "
            "date, amount, percentage, duration, and other numeric fact exactly.",
            prompt,
            max_tokens=256,
        )
        return _preserve_grounded_facts(prompt, response)


class GeminiLLM(BaseLLM):
    """Small Gemini REST adapter; keeps the demo installable without an SDK."""

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY to use Gemini mode.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.provider_name = f"Gemini ({self.model})"

    def _generate(self, instruction: str, prompt: str, max_tokens: int = 256) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
        }
        encoded_body = json.dumps(body).encode("utf-8")
        payload = None
        for attempt in range(3):
            request = urllib.request.Request(
                url,
                data=encoded_body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt < 2:
                    try:
                        delay = float(exc.headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        delay = 2 ** (attempt + 1)
                    time.sleep(min(max(delay, 1), 5))
                    continue
                if exc.code == 503:
                    raise RuntimeError(
                        "Gemini is temporarily unavailable or at capacity (HTTP 503). "
                        "Please retry in a little while or select another available model."
                    ) from exc
                raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail[:500]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Could not reach the Gemini API after 3 attempts: {exc}") from exc
        try:
            return "".join(
                part.get("text", "")
                for part in payload["candidates"][0]["content"]["parts"]
                if "text" in part
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned no text candidate.") from exc

    def classify_intent(self, text: str) -> str:
        labels = ", ".join(sorted(KNOWN_INTENTS))
        response = self._generate(
            f"Classify the user's customer-support request. Reply with exactly one label from: {labels}. "
            "A question about policy, fees, refunds, baggage, or rescheduling is policy_query, "
            "not a request to execute a cancellation.",
            text,
            max_tokens=32,
        )
        return _extract_intent(response)

    def classify_sentiment(self, text: str):
        response = self._generate(
            "Classify sentiment as positive, neutral, or negative and urgency as low, medium, or high. "
            "Reply only as sentiment,urgency (for example neutral,low).",
            text,
            max_tokens=16,
        )
        return _extract_sentiment_urgency(response)

    def generate_response(self, prompt: str) -> str:
        response = self._generate(
            "You are a concise customer-support agent. Use only the provided tool result or policy passage. "
            "Do not claim an action was completed unless the provided result confirms it. Preserve every booking ID, "
            "date, amount, percentage, duration, and other numeric fact exactly.",
            prompt,
            max_tokens=256,
        )
        return _preserve_grounded_facts(prompt, response)


def get_llm() -> BaseLLM:
    requested = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    if requested in {"", "auto"}:
        if os.environ.get("GEMINI_API_KEY"):
            requested = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            requested = "openai"
        else:
            requested = "rule"
    if requested in {"rule", "rulebased", "offline"}:
        return RuleBasedLLM()
    if requested == "ollama":
        return OllamaLLM()
    if requested == "gemini":
        return GeminiLLM()
    if requested == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAI mode.")
        return OpenAILLM()
    raise RuntimeError("LLM_PROVIDER must be auto, rule, ollama, gemini, or openai.")
