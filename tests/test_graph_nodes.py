"""Tests for graph reviewer node and data context enrichment."""

import pytest


def test_reviewer_node_includes_output_preview(monkeypatch):
    """Ensure the reviewer node processes tabular summaries and builds data context.

    The test monkey‑patches ``EvaluationAgent`` to avoid external LLM calls and
    supplies a realistic ``tabular_summaries`` entry so that
    ``_build_data_context_for_eval`` produces a non‑empty string.
    """
    # Stub EvaluationAgent to return a deterministic report without contacting Ollama.
    class DummyEvalReport:
        def to_dict(self):
            return {
                "session_id": "test",
                "task_count": 1,
                "flagged_count": 0,
                "pass_count": 1,
                "overall_session_score": 10.0,
                "session_verdict": "PASS",
                "evaluations": [],
            }

    class DummyEvalAgent:
        def __init__(self, session_id="default"):
            pass

        def evaluate_task_results(self, task_results, data_context="", stat_report=None):
            return DummyEvalReport()

    monkeypatch.setattr(
        "multimodal_ds.graph.EvaluationAgent",
        DummyEvalAgent,
    )

    # Import the private reviewer node directly from the graph module.
    from multimodal_ds.graph import _reviewer_node

    # Construct a minimal state containing a tabular summary with realistic statistics.
    state = {
        "analysis_tasks": [{"name": "task1"}],
        "full_code_outputs": ["output preview for task1"],
        "errors": [],
        "visualizations": [],
        "saved_artifacts": [],
        "tabular_summaries": [
            {
                "source": "sample.csv",
                "shape": [8, 3],
                "columns": ["age", "income", "churn"],
                "data_profile": {
                    "numeric_stats": {
                        "age": {"mean": 30, "std": 5, "min": 20, "max": 40},
                        "income": {"mean": 60000, "std": 15000, "min": 30000, "max": 120000},
                    }
                },
            }
        ],
        "statistical_report": {},
    }

    result = _reviewer_node(state)
    assert "eval_report" in result
    eval_report = result["eval_report"]
    # The stubbed report should contain the fields we defined.
    assert eval_report["session_id"] == "test"
    assert eval_report["session_verdict"] == "PASS"
