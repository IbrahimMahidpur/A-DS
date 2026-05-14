"""
ReportGeneratorAgent — produces structured executive, technical, and
business-recommendation documents from the final pipeline state.

Separation of concerns
-----------------------
- VisualizationAgent  → chart files (.png / .html)
- ReportGeneratorAgent → narrative documents (this file)
- reporter.py          → LangGraph node wrapper that calls both

GeneratedReport fields
-----------------------
  executive_summary        3-5 plain-prose sentences for a C-suite audience
  technical_report         Full markdown (methodology / results / quality)
  business_recommendations 3-7 actionable next steps
  confidence_score         0.0-1.0 derived from overall_session_score / 10
  risk_summary             Limitations and data-quality caveats
  model_artifacts          Saved .pkl / .joblib / .h5 filenames
  charts_referenced        Saved .html / .png filenames
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from multimodal_ds.config import (
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    OUTPUT_DIR,
    REVIEWER_MODEL,
)

logger = logging.getLogger(__name__)

# ── Output filename ────────────────────────────────────────────────────────────
_OUTPUT_FILENAME = "report_generator_output.json"

# ── Extension sets ─────────────────────────────────────────────────────────────
_CHART_EXTS  = {".html", ".png", ".jpg", ".jpeg", ".svg", ".webp"}
_MODEL_EXTS  = {".pkl", ".joblib", ".h5", ".pt", ".onnx"}
_DATA_EXTS   = {".csv", ".parquet", ".json", ".xlsx"}

# ── Fallback recommendations ───────────────────────────────────────────────────
_FALLBACK_RECS: List[str] = [
    "Review the full technical report for detailed methodology and results.",
    "Validate the trained model on an independent holdout dataset before deployment.",
    "Schedule quarterly model retraining to account for data drift.",
]


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class GeneratedReport:
    session_id: str
    executive_summary: str
    technical_report: str
    business_recommendations: List[str]
    confidence_score: float
    risk_summary: str
    model_artifacts: List[str] = field(default_factory=list)
    charts_referenced: List[str] = field(default_factory=list)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":                self.session_id,
            "executive_summary":         self.executive_summary,
            "technical_report":          self.technical_report,
            "business_recommendations":  self.business_recommendations,
            "confidence_score":          self.confidence_score,
            "risk_summary":              self.risk_summary,
            "model_artifacts":           self.model_artifacts,
            "charts_referenced":         self.charts_referenced,
        }

    def save(self, output_dir: Path) -> Path:
        """Serialise to JSON and return the written path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / _OUTPUT_FILENAME
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"[ReportGenerator] Saved → {out}")
        return out


# ── Agent ──────────────────────────────────────────────────────────────────────

class ReportGeneratorAgent:
    """
    Produces GeneratedReport from the final LangGraph state dict.

    Parameters
    ----------
    session_id :
        Used for working-directory isolation (OUTPUT_DIR / session_id).
    working_dir :
        Override the default output directory (useful for tests).
    """

    def __init__(
        self,
        session_id: str = "default",
        working_dir: Optional[str] = None,
    ) -> None:
        self.session_id  = session_id
        self.working_dir = (
            Path(working_dir) if working_dir else Path(OUTPUT_DIR) / session_id
        )
        self._model = REVIEWER_MODEL.replace("ollama/", "")

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate(self, state: dict) -> GeneratedReport:
        """
        Main entry point.  Extracts everything it needs from *state*, calls the
        three sub-generators, builds a GeneratedReport, persists it, and returns
        it.
        """
        user_query       = state.get("user_query", "")
        analysis_plan    = state.get("analysis_plan", "")
        code_outputs     = state.get("code_outputs", [])
        visualizations   = state.get("visualizations", [])
        saved_artifacts  = state.get("saved_artifacts", [])
        eval_report      = state.get("eval_report", {})
        errors           = state.get("errors", [])
        problem_spec     = state.get("problem_spec", {})

        if not isinstance(eval_report, dict):
            eval_report = (
                eval_report.to_dict()
                if hasattr(eval_report, "to_dict")
                else {}
            )

        # ── Classify artefacts ─────────────────────────────────────────────────
        all_files = list(visualizations) + list(saved_artifacts)
        charts_referenced = [
            f for f in all_files
            if Path(f).suffix.lower() in _CHART_EXTS
        ]
        model_artifacts = [
            f for f in all_files
            if Path(f).suffix.lower() in _MODEL_EXTS
        ]

        # ── Confidence score ───────────────────────────────────────────────────
        raw_score       = eval_report.get("overall_session_score", 5.0)
        confidence_score = max(0.0, min(1.0, float(raw_score) / 10.0))

        # ── Risk / limitations summary ─────────────────────────────────────────
        risk_summary = self._build_risk_summary(eval_report, errors)

        # ── LLM-driven sections ────────────────────────────────────────────────
        executive_summary = self._generate_executive_summary(
            user_query, code_outputs, eval_report, problem_spec
        )
        technical_report = self._generate_technical_report(state)
        business_recommendations = self._generate_recommendations(
            executive_summary, eval_report, problem_spec
        )

        report = GeneratedReport(
            session_id               = self.session_id,
            executive_summary        = executive_summary,
            technical_report         = technical_report,
            business_recommendations = business_recommendations,
            confidence_score         = confidence_score,
            risk_summary             = risk_summary,
            model_artifacts          = model_artifacts,
            charts_referenced        = charts_referenced,
        )

        report.save(self.working_dir)
        return report

    # ── Sub-generators ─────────────────────────────────────────────────────────

    def _generate_executive_summary(
        self,
        user_query: str,
        code_outputs: List[str],
        eval_report: dict,
        problem_spec: dict,
    ) -> str:
        """3-5 plain-prose sentences for a C-suite audience."""
        score  = eval_report.get("overall_session_score", 5.0)
        n_tasks = len(code_outputs)

        system = (
            "You are a business analyst writing for a C-suite audience. "
            "No jargon. No markdown. Plain prose only. 3-5 sentences maximum."
        )
        outputs_preview = "\n".join(code_outputs)[:3000]
        prompt = (
            f"The following data analysis was performed for the request:\n"
            f"'{user_query}'\n\n"
            f"Quality score: {score}/10 | Tasks executed: {n_tasks}\n\n"
            f"Outputs summary:\n{outputs_preview}\n\n"
            f"Write the executive summary now."
        )

        result = self._call_llm(prompt, system)
        if result:
            return result

        # Fallback — always non-empty
        return (
            f"Analysis of '{user_query}' completed with quality score "
            f"{score:.1f}/10. "
            f"{n_tasks} analytical task{'s' if n_tasks != 1 else ''} were executed. "
            f"See the technical report for detailed findings and methodology."
        )

    def _generate_technical_report(self, state: dict) -> str:
        """Full markdown report using the REPORTER_SYSTEM prompt from reporter.py."""
        # Lazy import to avoid circular dependency
        from multimodal_ds.agents.reporter import (  # noqa: PLC0415
            REPORTER_SYSTEM,
            _fallback_report,
            _call_ollama,
        )

        user_query    = state.get("user_query", "")
        plan          = state.get("analysis_plan", "No plan recorded.")
        all_outputs   = "\n\n".join(state.get("code_outputs", []))
        charts        = "\n".join(state.get("visualizations", []))
        artifacts     = "\n".join(state.get("saved_artifacts", []))
        errors        = "\n".join(state.get("errors", []))
        eval_report   = state.get("eval_report", {})
        if not isinstance(eval_report, dict):
            eval_report = {}

        eval_summary = ""
        if eval_report:
            verdict = eval_report.get("session_verdict", "UNKNOWN")
            score   = eval_report.get("overall_session_score", "N/A")
            eval_summary = f"Overall Quality Score: {score}/10 | Verdict: {verdict}"
            for ev in eval_report.get("evaluations", [])[:5]:
                eval_summary += (
                    f"\n- Task '{ev.get('task_name', '?')}': "
                    f"{ev.get('overall_score', '?')}/10"
                )

        prompt = (
            f"Original query: {user_query}\n\n"
            f"Analysis plan executed:\n{plan}\n\n"
            f"Step-by-step outputs:\n{all_outputs[:6000]}\n\n"
            f"Charts generated:\n{charts or 'None'}\n\n"
            f"Production Artifacts Saved:\n{artifacts or 'None'}\n\n"
            f"Evaluation summary:\n{eval_summary or 'No evaluation data'}\n\n"
            f"Errors encountered:\n{errors or 'None'}\n\n"
            "Write the complete analysis report now. Be sure to mention any "
            "saved models or data files in the Results or Recommendations section."
        )

        # Use the shared _call_ollama from reporter (same model + timeout)
        result = _call_ollama(prompt, system=REPORTER_SYSTEM)
        if result and not result.startswith("[Reporter] Ollama error"):
            return result
        return _fallback_report(prompt)

    def _generate_recommendations(
        self,
        executive_summary: str,
        eval_report: dict,
        problem_spec: dict,
    ) -> List[str]:
        """5 specific, actionable business recommendations as a list."""
        deliverables = problem_spec.get("required_deliverables", [])
        score = eval_report.get("overall_session_score", 5.0)

        system = (
            "You are a business strategy consultant. "
            "Given an analysis summary, generate exactly 5 specific, actionable "
            "business recommendations as a JSON array of strings. "
            "No preamble. Output ONLY the JSON array."
        )
        prompt = (
            f"Analysis summary:\n{executive_summary}\n\n"
            f"Quality score: {score}/10\n"
            f"Required deliverables: {deliverables}\n\n"
            "Output a JSON array of 5 actionable recommendations."
        )

        raw = self._call_llm(prompt, system)
        if raw:
            recs = self._parse_json_list(raw)
            if recs:
                return recs

        return _FALLBACK_RECS.copy()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: str) -> str:
        """
        Call Ollama; return content string on success, empty string on failure.
        Never raises.
        """
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": 2000, "temperature": 0.2},
                },
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                # Strip <think>…</think> reasoning traces
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                return content
            logger.warning(
                f"[ReportGenerator] Ollama HTTP {resp.status_code}"
            )
            return ""
        except Exception as exc:
            logger.warning(f"[ReportGenerator] LLM call failed: {exc}")
            return ""

    @staticmethod
    def _parse_json_list(raw: str) -> List[str]:
        """
        Extract a JSON list of strings from a raw LLM response.
        Handles markdown fences and trailing commas.
        Returns [] if parsing fails.
        """
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Find the first [...] block
        match = re.search(r"\[.*?\]", raw, flags=re.DOTALL)
        if not match:
            return []
        candidate = match.group(0)
        # Remove trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, list) and all(
                isinstance(s, str) for s in result
            ):
                return result
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _build_risk_summary(eval_report: dict, errors: List[str]) -> str:
        """Compose a limitations / caveats paragraph from eval data and errors."""
        parts: List[str] = []

        score = eval_report.get("overall_session_score", None)
        if score is not None and float(score) < 6:
            parts.append(
                f"Overall quality score is {score:.1f}/10, indicating potential "
                "reliability concerns. Results should be treated as indicative."
            )

        flagged = eval_report.get("flagged_count", 0)
        if flagged:
            parts.append(
                f"{flagged} task(s) were flagged during evaluation for quality issues."
            )

        if errors:
            parts.append(
                f"{len(errors)} error(s) were recorded during execution; "
                "some results may be incomplete."
            )

        if not parts:
            parts.append(
                "No significant limitations detected. Standard model assumptions apply."
            )

        return " ".join(parts)
