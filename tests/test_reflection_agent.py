"""
Tests for ReflectionAgent (Phase 5, Step 8b — Explicit Reflection Loop).
Run with: pytest tests/test_reflection_agent.py -v

All LLM calls are patched — tests run fully offline.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from multimodal_ds.core.message_bus import reset_bus
from multimodal_ds.core.context_pool import clear_context_pool


# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    """Reset bus + context pool between every test."""
    reset_bus()
    yield
    reset_bus()
    clear_context_pool("test_session")


def _make_eval_report(
    session_score: float = 5.0,
    dim_scores: dict | None = None,
    flagged: bool = True,
) -> dict:
    """Build a minimal eval_report dict matching EvaluationAgent.to_dict()."""
    _dims = {
        "statistical_validity": 5,
        "hallucination_risk": 5,
        "data_leakage": 5,
        "output_completeness": 5,
    }
    if dim_scores:
        _dims.update(dim_scores)

    dims_list = [
        {"name": k, "score": v, "reasoning": "test", "flagged": v < 4}
        for k, v in _dims.items()
    ]
    return {
        "session_id": "test_session",
        "overall_session_score": session_score,
        "evaluations": [
            {
                "task_name": "test_task",
                "overall_score": int(session_score),
                "flagged": flagged,
                "flag_reasons": ["test flag reason"],
                "dimensions": dims_list,
            }
        ],
    }


def _make_state(retry_count: int = 0) -> dict:
    return {"retry_count": retry_count}


def _agent(session_id: str = "test_session", max_retries: int = 3):
    from multimodal_ds.agents.reflection_agent import ReflectionAgent
    return ReflectionAgent(session_id=session_id, max_retries=max_retries)


# ══════════════════════════════════════════════════════════════════════════
#  ReflectionReport dataclass
# ══════════════════════════════════════════════════════════════════════════

class TestReflectionReport:
    def test_to_dict_roundtrip(self):
        from multimodal_ds.agents.reflection_agent import ReflectionReport
        report = ReflectionReport(
            failed_task_name="eda_task",
            root_cause="Model hallucinated a column",
            severity="HIGH",
            retry_strategy="Use only known columns",
            improved_instructions="Strictly use only columns listed in data_context",
            target_agent="executor",
            retry_count=1,
            max_retries_reached=False,
        )
        d = report.to_dict()
        assert d["failed_task_name"]      == "eda_task"
        assert d["root_cause"]            == "Model hallucinated a column"
        assert d["severity"]              == "HIGH"
        assert d["retry_strategy"]        == "Use only known columns"
        assert d["improved_instructions"] == "Strictly use only columns listed in data_context"
        assert d["target_agent"]          == "executor"
        assert d["retry_count"]           == 1
        assert d["max_retries_reached"]   is False

    def test_to_dict_all_keys_present(self):
        from multimodal_ds.agents.reflection_agent import ReflectionReport
        r = ReflectionReport(
            failed_task_name="t", root_cause="r", severity="LOW",
            retry_strategy="s", improved_instructions="i",
            target_agent="executor", retry_count=0, max_retries_reached=False,
        )
        keys = set(r.to_dict().keys())
        expected = {
            "failed_task_name", "root_cause", "severity", "retry_strategy",
            "improved_instructions", "target_agent", "retry_count",
            "max_retries_reached",
        }
        assert expected == keys

    def test_to_dict_serialisable_to_json(self):
        from multimodal_ds.agents.reflection_agent import ReflectionReport
        r = ReflectionReport(
            failed_task_name="t", root_cause="r", severity="CRITICAL",
            retry_strategy="s", improved_instructions="i",
            target_agent="planner", retry_count=3, max_retries_reached=True,
        )
        # Must not raise
        serialised = json.dumps(r.to_dict())
        restored   = json.loads(serialised)
        assert restored["severity"]            == "CRITICAL"
        assert restored["max_retries_reached"] is True


# ══════════════════════════════════════════════════════════════════════════
#  Rule-based fallback (LLM patched to raise)
# ══════════════════════════════════════════════════════════════════════════

class TestRuleBasedFallback:
    """All four dimension categories tested with LLM made unavailable."""

    def _reflect_with_dim(self, dim_name: str, bad_score: int = 2):
        agent  = _agent()
        report = _make_eval_report(
            session_score=4.0,
            dim_scores={dim_name: bad_score},
        )
        with patch("httpx.post", side_effect=Exception("Ollama offline")):
            return agent.reflect(eval_report=report, state=_make_state())

    def test_hallucination_risk_fallback(self):
        result = self._reflect_with_dim("hallucination_risk", bad_score=2)
        assert "column" in result.root_cause.lower() or "hallucin" in result.root_cause.lower() or "referenced" in result.root_cause.lower()
        assert "data_context" in result.improved_instructions or "columns" in result.improved_instructions.lower()

    def test_statistical_validity_fallback(self):
        result = self._reflect_with_dim("statistical_validity", bad_score=2)
        assert "statistic" in result.root_cause.lower() or "method" in result.root_cause.lower()
        assert "normality" in result.improved_instructions.lower() or "non-parametric" in result.improved_instructions.lower()

    def test_data_leakage_fallback(self):
        result = self._reflect_with_dim("data_leakage", bad_score=2)
        assert "target" in result.root_cause.lower() or "leakage" in result.root_cause.lower()
        assert "target" in result.improved_instructions.lower() or "exclude" in result.improved_instructions.lower()

    def test_output_completeness_fallback(self):
        result = self._reflect_with_dim("output_completeness", bad_score=2)
        assert "output" in result.root_cause.lower() or "file" in result.root_cause.lower()
        assert "savefig" in result.improved_instructions or "joblib" in result.improved_instructions

    def test_fallback_returns_reflection_report_type(self):
        from multimodal_ds.agents.reflection_agent import ReflectionReport
        result = self._reflect_with_dim("hallucination_risk")
        assert isinstance(result, ReflectionReport)

    def test_fallback_improved_instructions_non_empty(self):
        result = self._reflect_with_dim("statistical_validity")
        assert result.improved_instructions.strip() != ""

    def test_fallback_root_cause_non_empty(self):
        result = self._reflect_with_dim("data_leakage")
        assert result.root_cause.strip() != ""


# ══════════════════════════════════════════════════════════════════════════
#  Severity classification
# ══════════════════════════════════════════════════════════════════════════

class TestSeverityClassification:
    def _reflect_at_score(self, score: float):
        agent  = _agent()
        report = _make_eval_report(session_score=score)
        with patch("httpx.post", side_effect=Exception("offline")):
            return agent.reflect(eval_report=report, state=_make_state())

    def test_critical_when_score_below_3(self):
        result = self._reflect_at_score(2.0)
        assert result.severity == "CRITICAL"

    def test_critical_at_boundary_below_3(self):
        result = self._reflect_at_score(0.0)
        assert result.severity == "CRITICAL"

    def test_high_when_score_between_3_and_5(self):
        result = self._reflect_at_score(4.0)
        assert result.severity == "HIGH"

    def test_high_at_lower_boundary(self):
        result = self._reflect_at_score(3.0)
        assert result.severity == "HIGH"

    def test_medium_when_score_between_5_and_7(self):
        result = self._reflect_at_score(6.0)
        assert result.severity == "MEDIUM"

    def test_medium_at_lower_boundary(self):
        result = self._reflect_at_score(5.0)
        assert result.severity == "MEDIUM"

    def test_low_when_score_7_or_above(self):
        result = self._reflect_at_score(7.0)
        assert result.severity == "LOW"

    def test_low_at_high_score(self):
        result = self._reflect_at_score(9.5)
        assert result.severity == "LOW"


# ══════════════════════════════════════════════════════════════════════════
#  max_retries_reached flag
# ══════════════════════════════════════════════════════════════════════════

class TestMaxRetriesReached:
    def _reflect(self, retry_count: int, max_retries: int = 3):
        agent  = _agent(max_retries=max_retries)
        report = _make_eval_report(session_score=4.0)
        with patch("httpx.post", side_effect=Exception("offline")):
            return agent.reflect(
                eval_report=report,
                state=_make_state(retry_count=retry_count),
            )

    def test_not_reached_when_below_limit(self):
        result = self._reflect(retry_count=2, max_retries=3)
        assert result.max_retries_reached is False

    def test_reached_when_equal_to_limit(self):
        result = self._reflect(retry_count=3, max_retries=3)
        assert result.max_retries_reached is True

    def test_reached_when_above_limit(self):
        result = self._reflect(retry_count=5, max_retries=3)
        assert result.max_retries_reached is True

    def test_not_reached_at_zero(self):
        result = self._reflect(retry_count=0, max_retries=3)
        assert result.max_retries_reached is False

    def test_retry_count_propagated_correctly(self):
        result = self._reflect(retry_count=2, max_retries=3)
        assert result.retry_count == 2


# ══════════════════════════════════════════════════════════════════════════
#  target_agent routing
# ══════════════════════════════════════════════════════════════════════════

class TestTargetAgentRouting:
    def _reflect(self, session_score: float, retry_count: int):
        agent  = _agent(max_retries=3)
        report = _make_eval_report(session_score=session_score)
        with patch("httpx.post", side_effect=Exception("offline")):
            return agent.reflect(
                eval_report=report,
                state=_make_state(retry_count=retry_count),
            )

    def test_executor_for_low_severity(self):
        result = self._reflect(session_score=8.0, retry_count=0)
        assert result.target_agent == "executor"

    def test_executor_for_high_severity_low_retries(self):
        result = self._reflect(session_score=4.0, retry_count=1)
        assert result.target_agent == "executor"

    def test_planner_for_critical_at_retry_2(self):
        result = self._reflect(session_score=1.0, retry_count=2)
        assert result.target_agent == "planner"

    def test_planner_for_critical_above_retry_2(self):
        result = self._reflect(session_score=2.5, retry_count=4)
        assert result.target_agent == "planner"

    def test_executor_for_critical_at_retry_1(self):
        # CRITICAL but retry_count < 2 → still executor
        result = self._reflect(session_score=1.0, retry_count=1)
        assert result.target_agent == "executor"


# ══════════════════════════════════════════════════════════════════════════
#  LLM-powered diagnosis path
# ══════════════════════════════════════════════════════════════════════════

class TestLLMDiagnosis:
    def _llm_response(self, root_cause: str, strategy: str, instructions: str) -> MagicMock:
        payload = json.dumps({
            "root_cause":            root_cause,
            "retry_strategy":        strategy,
            "improved_instructions": instructions,
        })
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"message": {"content": payload}}
        return mock

    def test_llm_root_cause_used(self):
        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        mock   = self._llm_response(
            root_cause="Feature matrix contained target column",
            strategy="Remove target before encoding",
            instructions="Drop target column prior to StandardScaler",
        )
        with patch("httpx.post", return_value=mock):
            result = agent.reflect(eval_report=report, state=_make_state())
        assert "target" in result.root_cause.lower()

    def test_llm_improved_instructions_used(self):
        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        mock   = self._llm_response(
            root_cause="Wrong method",
            strategy="Switch test",
            instructions="Use Mann-Whitney U test instead of t-test",
        )
        with patch("httpx.post", return_value=mock):
            result = agent.reflect(eval_report=report, state=_make_state())
        assert "Mann-Whitney" in result.improved_instructions

    def test_falls_back_on_http_500(self):
        agent  = _agent()
        report = _make_eval_report(
            session_score=3.5,
            dim_scores={"hallucination_risk": 2},
        )
        mock = MagicMock()
        mock.status_code = 500
        with patch("httpx.post", return_value=mock):
            result = agent.reflect(eval_report=report, state=_make_state())
        # Rule-based should kick in
        assert result.root_cause != ""
        assert result.improved_instructions != ""

    def test_llm_response_with_think_tags_stripped(self):
        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        inner  = json.dumps({
            "root_cause":            "Columns invented",
            "retry_strategy":        "Restrict columns",
            "improved_instructions": "Use only listed columns",
        })
        content = f"<think>Let me reason...</think>{inner}"
        mock    = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"message": {"content": content}}
        with patch("httpx.post", return_value=mock):
            result = agent.reflect(eval_report=report, state=_make_state())
        assert result.root_cause == "Columns invented"


# ══════════════════════════════════════════════════════════════════════════
#  SharedContextPool storage
# ══════════════════════════════════════════════════════════════════════════

class TestContextPoolStorage:
    def test_reflection_report_stored_in_pool(self):
        from multimodal_ds.core.context_pool import get_context_pool
        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        with patch("httpx.post", side_effect=Exception("offline")):
            agent.reflect(eval_report=report, state=_make_state())
        pool   = get_context_pool("test_session")
        stored = pool.get("reflection_report")
        assert stored is not None
        assert stored["failed_task_name"] == "test_task"

    def test_reflection_report_severity_matches(self):
        from multimodal_ds.core.context_pool import get_context_pool
        agent  = _agent()
        report = _make_eval_report(session_score=2.0)  # CRITICAL
        with patch("httpx.post", side_effect=Exception("offline")):
            agent.reflect(eval_report=report, state=_make_state())
        pool   = get_context_pool("test_session")
        stored = pool.get("reflection_report")
        assert stored["severity"] == "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════
#  MessageBus CODE_RETRY publication
# ══════════════════════════════════════════════════════════════════════════

class TestCodeRetryPublication:
    def test_code_retry_published(self):
        from multimodal_ds.core.message_bus import get_bus, MessageType
        bus      = get_bus()
        received = []
        bus.subscribe(MessageType.CODE_RETRY, received.append)

        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        with patch("httpx.post", side_effect=Exception("offline")):
            agent.reflect(eval_report=report, state=_make_state())

        assert len(received) == 1

    def test_code_retry_has_high_priority(self):
        from multimodal_ds.core.message_bus import get_bus, MessageType, Priority
        bus      = get_bus()
        received = []
        bus.subscribe(MessageType.CODE_RETRY, received.append)

        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        with patch("httpx.post", side_effect=Exception("offline")):
            agent.reflect(eval_report=report, state=_make_state())

        assert received[0].priority == Priority.HIGH

    def test_code_retry_payload_contains_report(self):
        from multimodal_ds.core.message_bus import get_bus, MessageType
        bus      = get_bus()
        received = []
        bus.subscribe(MessageType.CODE_RETRY, received.append)

        agent  = _agent()
        report = _make_eval_report(session_score=4.0)
        with patch("httpx.post", side_effect=Exception("offline")):
            agent.reflect(eval_report=report, state=_make_state())

        payload = received[0].payload
        assert "reflection_report" in payload
        assert payload["reflection_report"]["failed_task_name"] == "test_task"


# ══════════════════════════════════════════════════════════════════════════
#  _classify_severity helper (unit tests)
# ══════════════════════════════════════════════════════════════════════════

class TestClassifySeverityHelper:
    def test_boundaries(self):
        from multimodal_ds.agents.reflection_agent import _classify_severity
        assert _classify_severity(0.0) == "CRITICAL"
        assert _classify_severity(2.9) == "CRITICAL"
        assert _classify_severity(3.0) == "HIGH"
        assert _classify_severity(4.9) == "HIGH"
        assert _classify_severity(5.0) == "MEDIUM"
        assert _classify_severity(6.9) == "MEDIUM"
        assert _classify_severity(7.0) == "LOW"
        assert _classify_severity(10.0) == "LOW"


# ══════════════════════════════════════════════════════════════════════════
#  _rule_based_diagnosis helper (unit tests)
# ══════════════════════════════════════════════════════════════════════════

class TestRuleBasedDiagnosisHelper:
    def _dims(self, **overrides):
        base = {
            "statistical_validity": 7,
            "hallucination_risk":   7,
            "data_leakage":         7,
            "output_completeness":  7,
        }
        base.update(overrides)
        return [{"name": k, "score": v} for k, v in base.items()]

    def test_hallucination_rule(self):
        from multimodal_ds.agents.reflection_agent import _rule_based_diagnosis
        cause, instr = _rule_based_diagnosis(
            "hallucination_risk",
            self._dims(hallucination_risk=2),
        )
        assert "column" in cause.lower() or "referenced" in cause.lower()
        assert "data_context" in instr

    def test_statistical_validity_rule(self):
        from multimodal_ds.agents.reflection_agent import _rule_based_diagnosis
        cause, instr = _rule_based_diagnosis(
            "statistical_validity",
            self._dims(statistical_validity=3),
        )
        assert "statistic" in cause.lower() or "method" in cause.lower()
        assert "normality" in instr.lower()

    def test_data_leakage_rule(self):
        from multimodal_ds.agents.reflection_agent import _rule_based_diagnosis
        cause, instr = _rule_based_diagnosis(
            "data_leakage",
            self._dims(data_leakage=1),
        )
        assert "target" in cause.lower()
        assert "target" in instr.lower() or "exclude" in instr.lower()

    def test_output_completeness_rule(self):
        from multimodal_ds.agents.reflection_agent import _rule_based_diagnosis
        cause, instr = _rule_based_diagnosis(
            "output_completeness",
            self._dims(output_completeness=2),
        )
        assert "output" in cause.lower() or "file" in cause.lower()
        assert "savefig" in instr or "joblib" in instr
