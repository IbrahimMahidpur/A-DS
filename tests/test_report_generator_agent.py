"""
Tests for ReportGeneratorAgent.

All tests are fully offline — httpx.post is patched at the module level so
no Ollama instance is required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multimodal_ds.agents.report_generator_agent import (
    GeneratedReport,
    ReportGeneratorAgent,
    _FALLBACK_RECS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_agent(tmp_path: Path) -> ReportGeneratorAgent:
    return ReportGeneratorAgent(session_id="test", working_dir=str(tmp_path))


def _minimal_state(**overrides) -> dict:
    base = {
        "user_query":       "Analyse customer churn",
        "analysis_plan":    "Step 1: EDA. Step 2: Model.",
        "code_outputs":     ["Accuracy: 0.87", "ROC-AUC: 0.91"],
        "visualizations":   [],
        "saved_artifacts":  [],
        "eval_report":      {"overall_session_score": 7.0, "session_verdict": "PASS"},
        "errors":           [],
        "problem_spec":     {},
    }
    base.update(overrides)
    return base


# ── 1. Executive summary fallback ──────────────────────────────────────────────

class TestExecutiveSummaryFallback:
    """When httpx raises, _generate_executive_summary must return a non-empty str."""

    def test_returns_nonempty_string_on_http_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            side_effect=ConnectionError("Ollama offline"),
        ):
            result = agent._generate_executive_summary(
                user_query="Churn prediction",
                code_outputs=["Accuracy: 0.87"],
                eval_report={"overall_session_score": 7.0},
                problem_spec={},
            )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_mentions_query(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            side_effect=OSError("timeout"),
        ):
            result = agent._generate_executive_summary(
                user_query="Revenue forecasting",
                code_outputs=[],
                eval_report={"overall_session_score": 5.0},
                problem_spec={},
            )
        assert "Revenue forecasting" in result


# ── 2. Recommendations fallback ────────────────────────────────────────────────

class TestRecommendationsFallback:
    """When httpx raises, _generate_recommendations must return a non-empty list."""

    def test_returns_list_with_items_on_http_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            side_effect=ConnectionError("Ollama offline"),
        ):
            recs = agent._generate_recommendations(
                executive_summary="The analysis is done.",
                eval_report={"overall_session_score": 6.0},
                problem_spec={},
            )
        assert isinstance(recs, list)
        assert len(recs) >= 1

    def test_fallback_matches_constant(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            side_effect=RuntimeError("broken"),
        ):
            recs = agent._generate_recommendations(
                executive_summary="Summary.",
                eval_report={},
                problem_spec={},
            )
        assert recs == _FALLBACK_RECS


# ── 3. Confidence score calculation ───────────────────────────────────────────

class TestConfidenceScore:
    """overall_session_score / 10 → confidence_score, clamped to [0, 1]."""

    @pytest.mark.parametrize("score,expected", [
        (8.0,  0.8),
        (10.0, 1.0),
        (0.0,  0.0),
        (5.0,  0.5),
        (3.5,  0.35),
    ])
    def test_confidence_from_score(
        self, tmp_path: Path, score: float, expected: float
    ) -> None:
        agent = _make_agent(tmp_path)
        state = _minimal_state(eval_report={"overall_session_score": score})

        # Patch httpx so no network calls happen
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            return_value=mock_resp,
        ), patch(
            "multimodal_ds.agents.reporter._call_ollama",
            return_value="# Report\nDone.",
        ):
            report = agent.generate(state)

        assert report.confidence_score == pytest.approx(expected, abs=1e-9)


# ── 4. Artefact classification ─────────────────────────────────────────────────

class TestArtifactExtraction:
    """Files should be bucketed into model_artifacts / charts_referenced correctly."""

    def test_mixed_artifacts(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        state = _minimal_state(
            saved_artifacts=["model.pkl", "data.csv"],
            visualizations=["plot.html"],
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            return_value=mock_resp,
        ), patch(
            "multimodal_ds.agents.reporter._call_ollama",
            return_value="# Report",
        ):
            report = agent.generate(state)

        assert report.model_artifacts   == ["model.pkl"]
        assert report.charts_referenced == ["plot.html"]
        # data.csv must NOT appear in either list
        assert "data.csv" not in report.model_artifacts
        assert "data.csv" not in report.charts_referenced

    def test_multiple_model_formats(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        state = _minimal_state(
            saved_artifacts=["model.pkl", "encoder.joblib", "deep.h5"],
            visualizations=["fig1.png", "fig2.html"],
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            return_value=mock_resp,
        ), patch(
            "multimodal_ds.agents.reporter._call_ollama",
            return_value="# Report",
        ):
            report = agent.generate(state)

        assert set(report.model_artifacts)   == {"model.pkl", "encoder.joblib", "deep.h5"}
        assert set(report.charts_referenced) == {"fig1.png", "fig2.html"}


# ── 5. save() writes report_generator_output.json ─────────────────────────────

class TestSave:
    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        report = GeneratedReport(
            session_id               = "s1",
            executive_summary        = "Done.",
            technical_report         = "# Report\nDetails.",
            business_recommendations = ["Rec 1", "Rec 2"],
            confidence_score         = 0.75,
            risk_summary             = "None.",
            model_artifacts          = ["model.pkl"],
            charts_referenced        = ["plot.html"],
        )
        out_path = report.save(tmp_path)

        assert out_path.name   == "report_generator_output.json"
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["session_id"]               == "s1"
        assert data["confidence_score"]         == 0.75
        assert data["business_recommendations"] == ["Rec 1", "Rec 2"]

    def test_save_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        report = GeneratedReport(
            session_id="s2",
            executive_summary="x",
            technical_report="y",
            business_recommendations=[],
            confidence_score=0.5,
            risk_summary="z",
        )
        out_path = report.save(nested)
        assert out_path.exists()

    def test_generate_calls_save(self, tmp_path: Path) -> None:
        """generate() must persist the file automatically."""
        agent = _make_agent(tmp_path)
        state = _minimal_state()

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch(
            "multimodal_ds.agents.report_generator_agent.httpx.post",
            return_value=mock_resp,
        ), patch(
            "multimodal_ds.agents.reporter._call_ollama",
            return_value="# Report",
        ):
            agent.generate(state)

        assert (tmp_path / "report_generator_output.json").exists()


# ── 6. _parse_json_list robustness ────────────────────────────────────────────

class TestParseJsonList:
    @pytest.mark.parametrize("raw,expected", [
        ('["A","B","C"]',                          ["A", "B", "C"]),
        ('```json\n["X","Y"]\n```',                ["X", "Y"]),
        ('Here you go:\n["P","Q","R",]',           ["P", "Q", "R"]),
        ("not a list",                             []),
        ("",                                       []),
    ])
    def test_various_formats(self, raw: str, expected: list) -> None:
        assert ReportGeneratorAgent._parse_json_list(raw) == expected
