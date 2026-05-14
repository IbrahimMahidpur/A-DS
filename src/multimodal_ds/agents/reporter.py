"""
Reporter Agent — synthesises all pipeline outputs into a structured
markdown report. This is the final node in the LangGraph StateGraph.

Design:
  - Reads code_outputs, visualizations, errors, eval_report, analysis_plan
  - Calls Ollama to produce a structured narrative report
  - Stores the report in state['final_report'] and writes it to disk
  - Also saves eval_report.json to the session working directory

Report structure:
  1. Executive Summary
  2. Key Findings (numbered, quantitative)
  3. Methodology
  4. Results (with inline chart references)
  5. Evaluation Quality Scores
  6. Limitations
  7. Recommendations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
import httpx

from multimodal_ds.config import REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.core.state import AgentState

logger = logging.getLogger(__name__)

REPORTER_SYSTEM = """You are a senior data scientist writing a final analysis report.

Structure your report EXACTLY as:
# Executive Summary
(2-3 sentences summarising what was done and the main finding)

# Key Findings
1. (quantitative finding with numbers)
2. ...

# Methodology
(what analysis steps were executed)

# Results
(detailed results, reference charts as ![Chart](filename.png))

# Quality Assessment
(summarise evaluation scores if provided)

# Limitations
(data quality issues, model assumptions, caveats)

# Recommendations
(3-5 actionable next steps)

Use markdown. Be precise and quantitative. Reference actual numbers from the outputs."""


def _call_ollama(prompt: str, system: str = REPORTER_SYSTEM) -> str:
    """Call Ollama reviewer model for report generation."""
    model = REVIEWER_MODEL.replace("ollama/", "")
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
                "options": {"num_predict": 4000, "temperature": 0.2},
            },
            timeout=LLM_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        return f"[Reporter] Ollama error: HTTP {response.status_code}"
    except Exception as e:
        logger.error(f"[Reporter] LLM call failed: {e}")
        return _fallback_report(prompt)


def _fallback_report(context: str) -> str:
    """Generate a minimal structured report when LLM is unavailable."""
    return f"""# Analysis Report

## Executive Summary
Analysis pipeline completed. LLM reporter was unavailable for narrative generation.
Raw outputs are included below for review.

## Raw Pipeline Outputs
```
{context[:3000]}
```

## Recommendations
- Review the raw code outputs above for key findings
- Re-run with the reporter LLM available for a structured narrative
"""


def reporter_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate the final report and save to disk.

    Reads:
      state['code_outputs'], state['visualizations'], state['errors'],
      state['analysis_plan'], state['eval_report'], state['user_query']

    Writes:
      state['final_report']  — markdown string
      {session_dir}/final_report.md  — on disk
      {session_dir}/eval_report.json — on disk
    """
    session_id = state.get("session_id", "default")
    logger.info(f"[Reporter] Generating final report for session {session_id}")

    # ── Assemble context for the LLM ──────────────────────────────────────
    all_outputs = "\n\n".join(state.get("code_outputs", []))
    charts = "\n".join(state.get("visualizations", []))
    artifacts = "\n".join(state.get("saved_artifacts", []))
    errors = "\n".join(state.get("errors", []))
    eval_report = state.get("eval_report", {})
    if not isinstance(eval_report, dict):
        # Support DummyReport objects
        eval_report = {
            "session_verdict": getattr(eval_report, "session_verdict", "UNKNOWN"),
            "overall_session_score": getattr(eval_report, "overall_session_score", "N/A"),
            "evaluations": getattr(eval_report, "evaluations", []),
        }
    plan = state.get("analysis_plan", "No plan recorded.")
    query = state.get("user_query", "")

    # Format evaluation scores
    eval_summary = ""
    if eval_report:
        verdict = eval_report.get("session_verdict", "UNKNOWN")
        score = eval_report.get("overall_session_score", "N/A")
        eval_summary = f"Overall Quality Score: {score}/10 | Verdict: {verdict}"
        evals = eval_report.get("evaluations", [])
        for ev in evals[:5]:
            eval_summary += f"\n- Task '{ev.get('task_name', '?')}': {ev.get('overall_score', '?')}/10"

    prompt = f"""Original query: {query}

Analysis plan executed:
{plan}

Step-by-step outputs:
{all_outputs[:6000]}

Charts generated:
{charts or 'None'}

Production Artifacts Saved:
{artifacts or 'None'}

Evaluation summary:
{eval_summary or 'No evaluation data'}

Errors encountered:
{errors or 'None'}

Write the complete analysis report now. Be sure to mention any saved models or data files in the Results or Recommendations section."""

    report = _call_ollama(prompt)

    # ── Delegate to ReportGeneratorAgent ──────────────────────────────────
    executive_summary       = ""
    business_recommendations: list = []
    try:
        from multimodal_ds.agents.report_generator_agent import ReportGeneratorAgent  # noqa: PLC0415
        rg_agent    = ReportGeneratorAgent(session_id=session_id)
        gen_report  = rg_agent.generate(state=state)
        executive_summary        = gen_report.executive_summary
        business_recommendations = gen_report.business_recommendations
        logger.info(
            f"[Reporter] ReportGeneratorAgent complete "
            f"(confidence={gen_report.confidence_score:.2f})"
        )
    except Exception as exc:
        logger.warning(f"[Reporter] ReportGeneratorAgent failed: {exc}")

    return {
        "final_report":             report,
        "executive_summary":        executive_summary,
        "business_recommendations": business_recommendations,
    }


# ── Standalone entry point (for direct calls from orchestrator) ────────────

def generate_report(
    user_query: str,
    analysis_plan: str,
    code_outputs: list[str],
    visualizations: list[str],
    errors: list[str],
    eval_report: dict,
    session_id: str,
    working_dir: Optional[str] = None,
) -> str:
    """
    Callable from AgentOrchestrator without going through the graph.
    Returns the markdown report string and saves it to disk.
    """
    state: AgentState = {
        "user_query": user_query,
        "uploaded_files": [],
        "_routing_flags": {},
        "parsed_documents": [],
        "image_embeddings": [],
        "audio_transcripts": [],
        "tabular_summaries": [],
        "statistical_report": {},
        "analysis_plan": analysis_plan,
        "analysis_tasks": [],
        "hypotheses": [],
        "current_step": 0,
        "steps_total": 0,
        "code_outputs": code_outputs,
        "visualizations": visualizations,
        "errors": errors,
        "retry_count": 0,
        "vector_store_id": "",
        "retrieved_context": "",
        "eval_report": eval_report,
        "final_report": "",
        "session_id": session_id,
        "messages": [],
    }

    result_state = reporter_agent(state)
    return result_state["final_report"]
