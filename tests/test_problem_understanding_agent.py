"""Tests for the ProblemUnderstandingAgent (Phase 1).

All LLM calls are mocked so the suite runs offline.
"""
import json
import pytest
from unittest.mock import MagicMock

# Ensure a clean bus for each test
@pytest.fixture(autouse=True)
def reset_message_bus():
    from multimodal_ds.core.message_bus import reset_bus
    reset_bus()
    yield
    reset_bus()


def test_heuristic_classify_classification():
    from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent
    agent = ProblemUnderstandingAgent()
    spec = agent._heuristic_classify("predict churn for next month", [])
    assert spec.task_type == "classification"
    assert spec.confidence == 0.3


def test_heuristic_classify_time_series():
    from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent
    agent = ProblemUnderstandingAgent()
    spec = agent._heuristic_classify("forecast revenue", [])
    assert spec.task_type == "time_series"
    assert spec.confidence == 0.3


def test_problem_spec_roundtrip():
    from multimodal_ds.agents.problem_understanding_agent import ProblemSpec
    original = ProblemSpec(
        task_type="regression",
        objective="Predict sales",
        constraints=["no external APIs"],
        required_deliverables=["trained model"],
        risk_flags=["small dataset"],
        confidence=0.85,
        ambiguities=["time horizon?"],
    )
    d = original.to_dict()
    restored = ProblemSpec.from_dict(d)
    assert original == restored


def test_understand_uses_llm(monkeypatch):
    # Mock httpx.post to return a predictable LLM response
    from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent, ProblemSpec
    from multimodal_ds.core.schema import UnifiedDocument

    # Prepare the JSON content the LLM would emit
    llm_dict = {
        "task_type": "classification",
        "objective": "Predict churn for next month",
        "constraints": ["no external APIs"],
        "required_deliverables": ["trained model"],
        "risk_flags": ["small dataset"],
        "confidence": 0.92,
        "ambiguities": []
    }
    llm_content = json.dumps(llm_dict)

    class DummyResponse:
        def __init__(self, content):
            self.status_code = 200
            self._content = content
        def json(self):
            return {"message": {"content": self._content}}

    def mock_post(*args, **kwargs):
        return DummyResponse(llm_content)

    monkeypatch.setattr("httpx.post", mock_post)

    agent = ProblemUnderstandingAgent(session_id="test_session")
    spec = agent.understand("any query", [UnifiedDocument()])

    assert isinstance(spec, ProblemSpec)
    assert spec.task_type == "classification"
    assert spec.objective == "Predict churn for next month"
    assert spec.confidence == 0.92
    # Ensure the spec was stored in the context pool
    from multimodal_ds.core.context_pool import get_context_pool
    pool = get_context_pool("test_session")
    stored = pool.get("problem_spec")
    assert stored["task_type"] == "classification"
    assert stored["confidence"] == 0.92
