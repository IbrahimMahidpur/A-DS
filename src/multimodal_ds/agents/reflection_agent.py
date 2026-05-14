"""
Reflection Agent — Phase 5, Step 8b: Explicit Reflection Loop.

Performs root-cause analysis on flagged evaluation reports, generates
improved task instructions, and publishes CODE_RETRY to the MessageBus so
the orchestration graph can route the retry to the correct agent.

Compared to the old _retry_node in graph.py (which only incremented a
counter), this agent:
  - Diagnoses *why* the task failed (LLM-powered or rule-based)
  - Classifies failure severity: LOW / MEDIUM / HIGH / CRITICAL
  - Rewrites the task instructions for the next attempt
  - Decides target_agent = "planner" on CRITICAL repeated failures
  - Stores a ReflectionReport in SharedContextPool
  - Publishes MessageType.CODE_RETRY with the full report payload

Bus events:
  CODE_RETRY (Priority.HIGH) — on every call to reflect()

Scoring dimensions read from eval_report (matches EvaluationAgent output):
  statistical_validity, hallucination_risk, data_leakage, output_completeness
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import httpx

from multimodal_ds.config import (
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    REVIEWER_MODEL,
)
from multimodal_ds.core.context_pool import get_context_pool

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_DIMENSION_THRESHOLD = 4          # score < this → dimension is problematic
_SEVERITY_CRITICAL   = 3          # overall_session_score < this → CRITICAL
_SEVERITY_HIGH       = 5          # overall_session_score < this → HIGH
_SEVERITY_MEDIUM     = 7          # overall_session_score < this → MEDIUM


# ══════════════════════════════════════════════════════════════════════════
#  ReflectionReport dataclass
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ReflectionReport:
    """Encapsulates root-cause analysis for a single failed task.

    Fields
    ------
    failed_task_name:
        Name of the task that was flagged by EvaluationAgent.
    root_cause:
        LLM-generated (or rule-based) explanation of what went wrong.
    severity:
        ``CRITICAL`` | ``HIGH`` | ``MEDIUM`` | ``LOW`` based on the
        session score at the time of reflection.
    retry_strategy:
        Short description of the approach for the next attempt.
    improved_instructions:
        Rewritten task description to send to the retry agent.
    target_agent:
        Which agent should handle the retry.  Almost always ``"executor"``
        unless the failure is CRITICAL *and* retries are exhausted, in
        which case ``"planner"`` is returned so the plan itself can be
        revised.
    retry_count:
        Number of retries that have already been attempted.
    max_retries_reached:
        ``True`` when ``retry_count >= max_retries``.
    """

    failed_task_name:      str
    root_cause:            str
    severity:              Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    retry_strategy:        str
    improved_instructions: str
    target_agent:          str
    retry_count:           int
    max_retries_reached:   bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_task_name":      self.failed_task_name,
            "root_cause":            self.root_cause,
            "severity":              self.severity,
            "retry_strategy":        self.retry_strategy,
            "improved_instructions": self.improved_instructions,
            "target_agent":          self.target_agent,
            "retry_count":           self.retry_count,
            "max_retries_reached":   self.max_retries_reached,
        }


# ══════════════════════════════════════════════════════════════════════════
#  ReflectionAgent
# ══════════════════════════════════════════════════════════════════════════

class ReflectionAgent:
    """Post-mortem analyser that produces actionable retry instructions.

    Usage::

        agent = ReflectionAgent(session_id="abc123", max_retries=3)
        report = agent.reflect(eval_report=eval_report.to_dict(), state=state)
        # report.improved_instructions → feed into next code-gen prompt
        # report.target_agent         → route retry to correct graph node
    """

    AGENT_NAME = "reflection_agent"

    _DIAGNOSIS_SYSTEM = (
        "You are a senior data science reviewer performing a post-mortem "
        "on a failed pipeline task. Respond ONLY with valid JSON — no "
        "markdown fences, no preamble, no explanation outside the JSON."
    )

    def __init__(
        self,
        session_id: str = "default",
        max_retries: int = 3,
    ) -> None:
        self.session_id  = session_id
        self.max_retries = max_retries
        self.pool        = get_context_pool(session_id)

    # ── Public API ─────────────────────────────────────────────────────────

    def reflect(self, eval_report: dict, state: dict) -> ReflectionReport:
        """Analyse *eval_report* and return a :class:`ReflectionReport`.

        Steps
        -----
        1. Extract the first flagged evaluation from ``eval_report["evaluations"]``.
        2. Identify the lowest-scoring dimension to find the root-cause
           category.
        3. Ask Ollama (REVIEWER_MODEL) for a diagnosis; fall back to
           rule-based heuristics if the LLM is unavailable.
        4. Classify severity from ``overall_session_score``.
        5. Choose ``target_agent`` based on severity + retry count.
        6. Persist report in SharedContextPool and publish CODE_RETRY.

        Parameters
        ----------
        eval_report:
            Serialised :class:`~multimodal_ds.agents.evaluation_agent.EvalReport`
            dict (use ``eval_report.to_dict()``).
        state:
            Current LangGraph state dict — used to read ``retry_count``.
        """
        # ── 1. Find the first flagged task ──────────────────────────────
        evaluations: list = eval_report.get("evaluations", [])
        flagged = [e for e in evaluations if e.get("flagged")]
        target_eval: dict = flagged[0] if flagged else (evaluations[0] if evaluations else {})

        failed_task_name: str = target_eval.get("task_name", "unknown_task")
        dimensions: list      = target_eval.get("dimensions", [])

        # ── 2. Identify lowest-scoring dimension ────────────────────────
        worst_dim: Optional[str] = _lowest_dimension(dimensions)

        # ── 3. Derive retry metadata from state ─────────────────────────
        retry_count: int      = int(state.get("retry_count", 0))
        max_retries_reached   = retry_count >= self.max_retries

        # ── 4. Classify severity from session score ──────────────────────
        session_score: float  = float(eval_report.get("overall_session_score", 5.0))
        severity = _classify_severity(session_score)

        # ── 5. Choose target_agent ───────────────────────────────────────
        if severity == "CRITICAL" and retry_count >= 2:
            target_agent = "planner"
        else:
            target_agent = "executor"

        # ── 6. LLM diagnosis (falls back to rule-based) ─────────────────
        llm_result = self._call_llm_diagnosis(
            task_name=failed_task_name,
            target_eval=target_eval,
            worst_dim=worst_dim,
        )

        if llm_result:
            root_cause            = llm_result.get("root_cause", "")
            improved_instructions = llm_result.get("improved_instructions", "")
            retry_strategy        = llm_result.get("retry_strategy", "LLM-guided retry")
        else:
            # Rule-based fallback keyed on worst dimension
            root_cause, improved_instructions = _rule_based_diagnosis(
                worst_dim, dimensions
            )
            retry_strategy = f"Rule-based retry for {worst_dim or 'unknown'} failure"

        # Guarantee non-empty strings
        if not root_cause:
            root_cause = "Unknown failure — LLM and rule-based diagnosis both unavailable"
        if not improved_instructions:
            improved_instructions = "Re-run the task with stricter validation checks"

        # ── 7. Build report ──────────────────────────────────────────────
        report = ReflectionReport(
            failed_task_name=failed_task_name,
            root_cause=root_cause,
            severity=severity,
            retry_strategy=retry_strategy,
            improved_instructions=improved_instructions,
            target_agent=target_agent,
            retry_count=retry_count,
            max_retries_reached=max_retries_reached,
        )

        # ── 8. Persist and publish ───────────────────────────────────────
        self.pool.set("reflection_report", report.to_dict(), agent=self.AGENT_NAME)
        self._publish_code_retry(report)

        logger.info(
            "[ReflectionAgent] session=%s task=%r severity=%s target=%s "
            "retry=%d/%d max_reached=%s",
            self.session_id,
            failed_task_name,
            severity,
            target_agent,
            retry_count,
            self.max_retries,
            max_retries_reached,
        )
        return report

    # ── LLM diagnosis ──────────────────────────────────────────────────────

    def _call_llm_diagnosis(
        self,
        task_name: str,
        target_eval: dict,
        worst_dim: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Ask the REVIEWER_MODEL to diagnose the failure.

        Returns a dict with ``root_cause``, ``retry_strategy``, and
        ``improved_instructions`` keys, or ``None`` on any failure.
        """
        prompt = _build_diagnosis_prompt(task_name, target_eval, worst_dim)
        model  = REVIEWER_MODEL.replace("ollama/", "")

        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":   model,
                    "messages": [
                        {"role": "system", "content": self._DIAGNOSIS_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 400},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code != 200:
                logger.warning(
                    "[ReflectionAgent] LLM returned HTTP %d", response.status_code
                )
                return None

            content = response.json().get("message", {}).get("content", "").strip()
            return _parse_diagnosis_response(content)

        except Exception as exc:
            logger.warning("[ReflectionAgent] LLM call failed: %s", exc)
            return None

    # ── Bus helper ─────────────────────────────────────────────────────────

    def _publish_code_retry(self, report: ReflectionReport) -> None:
        try:
            from multimodal_ds.core.message_bus import (
                AgentMessage,
                MessageType,
                Priority,
                get_bus,
            )
            get_bus().publish(AgentMessage(
                msg_type=MessageType.CODE_RETRY,
                payload={
                    "session_id":         self.session_id,
                    "reflection_report":  report.to_dict(),
                },
                sender=self.AGENT_NAME,
                session_id=self.session_id,
                priority=Priority.HIGH,
            ))
        except Exception as exc:
            logger.warning("[ReflectionAgent] Could not publish CODE_RETRY: %s", exc)


# ══════════════════════════════════════════════════════════════════════════
#  Module-level helpers (pure functions — easy to unit test)
# ══════════════════════════════════════════════════════════════════════════

def _classify_severity(
    session_score: float,
) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """Map an overall session score (0–10) to a severity label."""
    if session_score < _SEVERITY_CRITICAL:
        return "CRITICAL"
    if session_score < _SEVERITY_HIGH:
        return "HIGH"
    if session_score < _SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _lowest_dimension(dimensions: list) -> Optional[str]:
    """Return the name of the dimension with the lowest score, or ``None``."""
    best: Optional[tuple] = None          # (score, name)
    for dim in dimensions:
        try:
            score = int(dim.get("score", 10))
            name  = str(dim.get("name", ""))
        except (TypeError, ValueError):
            continue
        if best is None or score < best[0]:
            best = (score, name)
    return best[1] if best else None


def _rule_based_diagnosis(
    worst_dim: Optional[str],
    dimensions: list,
) -> tuple[str, str]:
    """Return ``(root_cause, improved_instructions)`` using hand-crafted rules.

    Falls back when the LLM is unavailable.  The rules match the four
    evaluation dimensions and their threshold score of 4.
    """
    # Build a score look-up from the dimensions list
    scores: Dict[str, int] = {}
    for dim in dimensions:
        try:
            scores[dim["name"]] = int(dim["score"])
        except (KeyError, TypeError, ValueError):
            pass

    # Evaluate each dimension in priority order
    hall  = scores.get("hallucination_risk",  10)
    stat  = scores.get("statistical_validity", 10)
    leak  = scores.get("data_leakage",         10)
    compl = scores.get("output_completeness",  10)

    if hall < _DIMENSION_THRESHOLD:
        return (
            "Model referenced columns not in dataset",
            "Strictly use only columns listed in data_context",
        )
    if stat < _DIMENSION_THRESHOLD:
        return (
            "Inappropriate statistical method for data distribution",
            "Check normality first; use non-parametric tests if violated",
        )
    if leak < _DIMENSION_THRESHOLD:
        return (
            "Target variable used during feature engineering",
            "Exclude target column before any scaling or encoding",
        )
    if compl < _DIMENSION_THRESHOLD:
        return (
            "Expected output files not produced",
            "Ensure all plt.savefig() and joblib.dump() calls execute before script ends",
        )

    # Generic fallback when no dimension is critically low
    return (
        f"Degraded quality in {worst_dim or 'unknown'} dimension",
        "Re-run task with additional validation and logging",
    )


def _build_diagnosis_prompt(
    task_name: str,
    target_eval: dict,
    worst_dim: Optional[str],
) -> str:
    dimensions_json = json.dumps(target_eval.get("dimensions", []), indent=2)
    flag_reasons    = target_eval.get("flag_reasons", [])
    overall_score   = target_eval.get("overall_score", "?")

    return f"""A data science pipeline task has been flagged for low quality.

Task name:     {task_name}
Overall score: {overall_score}/10
Worst dimension: {worst_dim or "unknown"}
Flag reasons:
{json.dumps(flag_reasons, indent=2)}

Dimension scores:
{dimensions_json}

Based on the above, diagnose the root cause and provide improved task instructions.

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "root_cause": "<one-sentence explanation of why the task failed>",
  "retry_strategy": "<brief description of the retry approach>",
  "improved_instructions": "<rewritten task description for the retry agent>"
}}"""


def _parse_diagnosis_response(content: str) -> Optional[Dict[str, Any]]:
    """Extract and validate the JSON diagnosis from a raw LLM response."""
    # Strip <think>...</think> blocks (some reasoning models include these)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Strip optional markdown fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1)

    # Locate first { ... } block
    start = content.find("{")
    if start == -1:
        return None

    depth, end = 0, -1
    for i, ch in enumerate(content[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    raw = content[start:end]
    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([\]}])", r"\1", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    required = {"root_cause", "retry_strategy", "improved_instructions"}
    if not required.issubset(parsed.keys()):
        return None

    return parsed
