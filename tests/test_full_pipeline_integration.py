"""
tests/test_full_pipeline_integration.py
========================================
Smoke test that verifies the 12 LangGraph nodes wire together correctly
without any real LLM or external service.

Strategy
--------
- httpx.post is patched to always raise ConnectionError — every agent falls
  back to its heuristic / offline path.
- Presidio is pre-stubbed in tests/conftest.py (runs before this file imports
  multimodal_ds.graph).
- MessageBus is reset around the bus test so no state bleeds between tests.
- No Ollama instance, no internet, no file-system fixtures beyond tmp_path.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

# ── Module-level httpx patch ───────────────────────────────────────────────────
# Applied as a module-scoped autouse fixture so every test in this file
# automatically runs without any real HTTP calls.

@pytest.fixture(autouse=True, scope="module")
def _block_httpx():
    """Raise ConnectionError on every httpx.post call for the whole module."""
    with patch("httpx.post", side_effect=ConnectionError("Ollama offline (test)")):
        yield


# ── Shared minimal-state fixture ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def complete_pipeline_state():
    """
    A minimal but valid state dict that matches make_initial_state() output,
    including all fields added in Phase 5 (problem_spec, reflection_report,
    executive_summary, business_recommendations).
    """
    return {
        # Core identity
        "session_id":              "integration_test",
        "user_query":              "analyse customer churn",
        # File uploads
        "uploaded_files":          [],
        "_routing_flags":          {},
        # Ingestion outputs
        "parsed_documents":        [],
        "image_embeddings":        [],
        "audio_transcripts":       [],
        "tabular_summaries":       [],
        # Planning / execution
        "analysis_plan":           "",
        "analysis_tasks":          [],
        "hypotheses":              [],
        "current_step":            0,
        "steps_total":             0,
        "code_outputs":            [],
        "full_code_outputs":       [],
        "visualizations":          [],
        "saved_artifacts":         [],
        "errors":                  [],
        "reflections":             [],
        # Quality / retry
        "retry_count":             0,
        "eval_report":             {},
        "reflection_report":       {},
        "gate_passed":             False,
        "gate_reasons":            [],
        # Memory / retrieval
        "vector_store_id":         "",
        "retrieved_context":       "",
        # Reports
        "final_report":            "",
        "executive_summary":       "",
        "business_recommendations": [],
        # Problem spec (Phase 5)
        "problem_spec":            {},
        # Additional fields
        "statistical_report":      {},
        "model_selection":         {},
        "ensemble_code_template":  "",
        "tuning_results":          {},
        "problem_spec":            {},
        "_last_task_name":         "",
        "_last_files_created":     [],
        "_last_success":           False,
        "current_step_files":      [],
        "current_step_success":    False,
        "_files_per_step":         [],
        "messages":                [],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1 — Problem understanding node runs offline
# ══════════════════════════════════════════════════════════════════════════════

def test_problem_understanding_node_runs_without_llm(complete_pipeline_state):
    """
    Node must return a problem_spec dict with a recognised task_type even when
    the LLM is unreachable (heuristic fallback path).
    """
    from multimodal_ds.graph import _problem_understanding_node

    result = _problem_understanding_node(
        {**complete_pipeline_state, "user_query": "predict customer churn"}
    )

    assert "problem_spec" in result, "problem_spec key missing from node output"
    spec = result["problem_spec"]
    assert isinstance(spec, dict), "problem_spec must be a dict"
    task_type = spec.get("task_type")
    valid_types = {"classification", "regression", "clustering",
                   "time_series", "nlp", "multimodal", "unknown"}
    assert task_type in valid_types, (
        f"task_type {task_type!r} not in {valid_types}"
    )
    # 'churn' matches the classification heuristic pattern
    assert task_type in ("classification", "unknown"), (
        f"Expected 'classification' (or 'unknown') for 'predict customer churn', "
        f"got {task_type!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2 — Router node sets table flag for a .csv file
# ══════════════════════════════════════════════════════════════════════════════

def test_router_node_flags_correct_type(complete_pipeline_state, tmp_path):
    """
    When a CSV file is present in uploaded_files, _routing_flags['table']
    must be True.
    """
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("age,income,churn\n25,45000,0\n30,62000,1\n")

    from multimodal_ds.graph import _router_node

    state = {**complete_pipeline_state, "uploaded_files": [str(csv_file)]}
    result = _router_node(state)

    assert "_routing_flags" in result
    flags = result["_routing_flags"]
    assert flags.get("table") is True, (
        f"Expected _routing_flags['table'] = True, got {flags}"
    )
    # non-CSV types must be False (file is .csv only)
    assert flags.get("doc")   is False
    assert flags.get("image") is False
    assert flags.get("audio") is False


# ══════════════════════════════════════════════════════════════════════════════
#  Test 3 — Reflection node enriches failed task description
# ══════════════════════════════════════════════════════════════════════════════

def test_reflection_node_enriches_failed_task(complete_pipeline_state):
    """
    _retry_node must:
      - increment retry_count by 1
      - rewind current_step to max(0, current_step - 1)
      - inject 'REFLECTION GUIDANCE' into the failed task's description
    All via the heuristic fallback (LLM offline).
    """
    from multimodal_ds.graph import _retry_node

    state = {
        **complete_pipeline_state,
        "analysis_tasks": [{"name": "EDA", "description": "Run EDA"}],
        "current_step":   1,
        "retry_count":    0,
        "eval_report": {
            "overall_session_score": 2.0,
            "flagged_count":         1,
            "evaluations": [{
                "task_name":    "EDA",
                "overall_score": 2,
                "flagged":       True,
                "dimensions": [
                    {"name": "hallucination_risk",    "score": 2,
                     "reasoning": "Referenced missing column"},
                    {"name": "statistical_validity",  "score": 5,
                     "reasoning": "ok"},
                    {"name": "data_leakage",          "score": 5,
                     "reasoning": "ok"},
                    {"name": "output_completeness",   "score": 5,
                     "reasoning": "ok"},
                ],
                "flag_reasons": ["hallucination_risk scored 2/10"],
            }],
        },
    }

    result = _retry_node(state)

    # retry_count incremented
    assert result["retry_count"] == 1, (
        f"retry_count should be 1, got {result['retry_count']}"
    )
    # current_step rewound by 1 (1 → 0)
    assert result["current_step"] == 0, (
        f"current_step should be 0 (rewound), got {result['current_step']}"
    )
    # REFLECTION GUIDANCE injected into task description
    task_desc = result["analysis_tasks"][0]["description"]
    assert "REFLECTION GUIDANCE" in task_desc, (
        f"Expected 'REFLECTION GUIDANCE' in task description, got:\n{task_desc}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 4 — make_initial_state contains all required keys
# ══════════════════════════════════════════════════════════════════════════════

def test_state_has_all_required_keys_after_initial_state():
    """
    make_initial_state() must include every key required by the full pipeline,
    including the Phase-5 additions.
    """
    from multimodal_ds.graph import make_initial_state

    state = make_initial_state(user_query="test query", uploaded_files=[])

    required_keys = [
        "user_query",
        "uploaded_files",
        "problem_spec",
        "reflection_report",
        "executive_summary",
        "business_recommendations",
        "session_id",
        "analysis_tasks",
        "eval_report",
        "final_report",
    ]
    for key in required_keys:
        assert key in state, f"make_initial_state() is missing key: {key!r}"

    # Spot-check types of new fields
    assert isinstance(state["executive_summary"],       str),  "executive_summary must be str"
    assert isinstance(state["business_recommendations"], list), "business_recommendations must be list"
    assert isinstance(state["problem_spec"],            dict), "problem_spec must be dict"
    assert isinstance(state["reflection_report"],       dict), "reflection_report must be dict"


# ══════════════════════════════════════════════════════════════════════════════
#  Test 5 — MessageBus receives PLAN_REQUEST from ProblemUnderstandingAgent
# ══════════════════════════════════════════════════════════════════════════════

def test_message_bus_receives_events_from_new_agents():
    """
    ProblemUnderstandingAgent.understand() must publish a PLAN_REQUEST message
    to the MessageBus even when the LLM is unreachable (heuristic fallback).
    """
    from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
    from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent

    reset_bus()
    bus = get_bus()

    received_types: list[MessageType] = []
    bus.subscribe_all(lambda m: received_types.append(m.msg_type))

    agent = ProblemUnderstandingAgent(session_id="integration_test")
    agent.understand(user_query="classify customer segments", documents=[])

    assert MessageType.PLAN_REQUEST in received_types, (
        f"Expected PLAN_REQUEST in bus events, got: {[t.value for t in received_types]}"
    )

    # Cleanup — don't let subscriptions bleed into other tests
    reset_bus()


# ══════════════════════════════════════════════════════════════════════════════
#  Test 6 — Router node handles empty uploaded_files gracefully
# ══════════════════════════════════════════════════════════════════════════════

def test_router_node_all_flags_false_when_no_files(complete_pipeline_state):
    """No uploaded files → all routing flags must be False."""
    from multimodal_ds.graph import _router_node

    state = {**complete_pipeline_state, "uploaded_files": []}
    result = _router_node(state)

    flags = result["_routing_flags"]
    for flag_name, flag_val in flags.items():
        assert flag_val is False, (
            f"Expected _routing_flags[{flag_name!r}] = False, got {flag_val}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 7 — Reflection node handles zero-task state without crashing
# ══════════════════════════════════════════════════════════════════════════════

def test_reflection_node_handles_empty_task_list(complete_pipeline_state):
    """
    _retry_node must not raise when analysis_tasks is empty.
    It should still increment retry_count.
    """
    from multimodal_ds.graph import _retry_node

    state = {
        **complete_pipeline_state,
        "analysis_tasks": [],
        "current_step":   0,
        "retry_count":    0,
        "eval_report":    {"overall_session_score": 3.0},
    }

    result = _retry_node(state)
    assert result["retry_count"] == 1
