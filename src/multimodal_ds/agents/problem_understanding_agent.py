"""
Problem Understanding Agent – Phase 1 of the orchestration pipeline.

This agent analyses a user query (and any already‑ingested UnifiedDocument objects)
and produces a structured ``ProblemSpec`` describing the intended data‑science task.
It is the first step before any ingestion or modelling.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal

from multimodal_ds.config import PLANNER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT
from multimodal_ds.core.schema import UnifiedDocument, DataType
from multimodal_ds.core.context_pool import get_context_pool
from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions – JSON extraction & Ollama call (mirrored from planner_agent)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract the outermost JSON object from LLM output.
    Handles optional <think> blocks, markdown fences, and prefers a JSON object over a list.
    Returns the raw JSON substring.
    """
    # Strip any <think>...</think> blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # If a fenced JSON block is present, extract its contents
    fence_match = re.search(r'```(?:json)?\s*([\[{].*?[\]}])\s*```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    # Prefer JSON object (starting with '{')
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return text[start:]
    # Fallback: look for a JSON list
    start = text.find('[')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return text[start:]
    # No JSON found – return the cleaned text
    return text


def _call_ollama(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Call Ollama's chat endpoint using the planner model.

    Returns the raw ``content`` string from the response, or an error placeholder.
    """
    import httpx
    model = PLANNER_MODEL.replace("ollama/", "")
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            },
            timeout=LLM_TIMEOUT,
        )
        if response.status_code == 200:
            resp_json = response.json()
            if isinstance(resp_json, dict):
                return resp_json.get("message", {}).get("content", "")
            return str(resp_json)
        return f"[Error: HTTP {response.status_code}]"
    except Exception as e:
        return f"[Error: {e}]"

# ---------------------------------------------------------------------------
# Data model – ProblemSpec
# ---------------------------------------------------------------------------

TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "time_series",
    "nlp",
    "multimodal",
    "unknown",
]

@dataclass
class ProblemSpec:
    task_type: TaskType
    objective: str
    constraints: List[str] = field(default_factory=list)
    required_deliverables: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ambiguities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to plain‑dict for LangGraph checkpointing."""
        return {
            "task_type": self.task_type,
            "objective": self.objective,
            "constraints": list(self.constraints),
            "required_deliverables": list(self.required_deliverables),
            "risk_flags": list(self.risk_flags),
            "confidence": float(self.confidence),
            "ambiguities": list(self.ambiguities),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec":
        return cls(
            task_type=d.get("task_type", "unknown"),
            objective=d.get("objective", ""),
            constraints=d.get("constraints", []),
            required_deliverables=d.get("required_deliverables", []),
            risk_flags=d.get("risk_flags", []),
            confidence=d.get("confidence", 0.0),
            ambiguities=d.get("ambiguities", []),
        )

# ---------------------------------------------------------------------------
# ProblemUnderstandingAgent implementation
# ---------------------------------------------------------------------------

class ProblemUnderstandingAgent:
    """Phase 1 – infer the problem specification from user intent and docs."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.logger = logger

    # Public API -----------------------------------------------------------
    def understand(self, user_query: str, documents: List[UnifiedDocument]) -> ProblemSpec:
        """Derive a ``ProblemSpec`` using LLM if possible, otherwise heuristics.

        The result is stored in the shared context pool under ``problem_spec`` and a
        ``PLAN_REQUEST`` message is published on the bus.
        """
        # Try LLM first
        try:
            doc_types = [doc.data_type.value for doc in documents]
            doc_summary = ", ".join(doc_types) if doc_types else "none"
            prompt = (
                "You are a senior data scientist tasked with formalising a user request."
                " Provide a JSON object with the following fields: task_type, objective, constraints,"
                " required_deliverables, risk_flags, confidence, ambiguities.\n"
                f"User query: {user_query}\n"
                f"Document types available: {doc_summary}\n"
                "Respond ONLY with valid JSON, no additional text."
            )
            raw = _call_ollama(prompt)
            cleaned = _extract_json(raw)
            spec_dict = json.loads(cleaned)
            spec = ProblemSpec.from_dict(spec_dict)
            self.logger.info("[ProblemUnderstanding] LLM derived spec: %s", spec.task_type)
        except Exception as e:
            self.logger.warning("[ProblemUnderstanding] LLM failed (%s); falling back to heuristic", e)
            spec = self._heuristic_classify(user_query, documents)

        # Store in shared context pool
        pool = get_context_pool(self.session_id)
        pool.set("problem_spec", spec.to_dict(), agent="problem_understanding_agent")

        # Publish a PLAN_REQUEST so downstream agents know a new problem is defined
        try:
            bus = get_bus()
            bus.publish(
                AgentMessage(
                    msg_type=MessageType.PLAN_REQUEST,
                    payload={"problem_spec": spec.to_dict()},
                    sender="problem_understanding_agent",
                    session_id=self.session_id,
                )
            )
        except Exception as pub_err:
            self.logger.error("[ProblemUnderstanding] Failed to publish PLAN_REQUEST: %s", pub_err)

        return spec

    # -------------------------------------------------------------------
    def _heuristic_classify(self, query: str, documents: List[UnifiedDocument]) -> ProblemSpec:
        """Fallback classifier based on regex patterns.

        Returns a ``ProblemSpec`` with low confidence (0.3) and minimal inference.
        """
        lowered = query.lower()
        task_type: TaskType = "unknown"
        # Order matters – check more specific patterns first
        patterns = [
            ("time_series", r"\b(forecast|time series|trend|seasonal)\b"),
            ("classification", r"\b(classify|classification|predict.*\b(class|label|binary)\b|churn)\b"),
            ("regression", r"\b(predict|forecast|estimate|price|sales|revenue|duration)\b"),
            ("clustering", r"\b(cluster|segmentation|group|grouping)\b"),
            ("nlp", r"\b(text|nlp|sentiment|language|entity|token)\b"),
            ("multimodal", r"\b(multimodal|image.*text|audio.*text|video)\b"),
        ]
        for ttype, pat in patterns:
            if re.search(pat, lowered):
                task_type = ttype  # type: ignore[assignment]
                break

        # Infer required deliverables from document data types
        deliverables: List[str] = []
        data_types = {doc.data_type for doc in documents}
        if DataType.TABULAR in data_types:
            deliverables.extend(["trained model", "feature importance chart"])
        if DataType.TEXT in data_types:
            deliverables.append("executive report")
        if DataType.IMAGE in data_types:
            deliverables.append("visualization chart")
        if DataType.AUDIO in data_types:
            deliverables.append("audio summary")

        spec = ProblemSpec(
            task_type=task_type,
            objective=query,
            constraints=[],
            required_deliverables=deliverables,
            risk_flags=[],
            confidence=0.3,
            ambiguities=[],
        )
        self.logger.info("[ProblemUnderstanding] Heuristic spec: %s (confidence %.2f)", spec.task_type, spec.confidence)
        return spec

