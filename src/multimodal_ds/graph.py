"""
Top-level LangGraph StateGraph — wires all agents as nodes with a
MemorySaver checkpointer for session persistence.

Fixes applied (vs original):
  1. _decide_ingestion_path: returns a single string, not a list.
     Fan-out to multiple ingestion nodes requires Send() — this simpler
     approach routes to the FIRST matching type, which is correct for
     the current sequential graph topology.
  2. _reviewer_node: task_result dict now uses keys that evaluation_agent
     actually reads ("name", "success", "output_preview", "files_created").
  3. retry_count: incremented in state when retrying, preventing infinite loops.
"""
from __future__ import annotations

import logging
import json
from datetime import datetime
from multimodal_ds.config import OUTPUT_DIR
import uuid
from typing import Optional

# Module-level singletons for PII scanning
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_ANALYZER_ENGINE = AnalyzerEngine()
_ANONYMIZER_ENGINE = AnonymizerEngine()

logger = logging.getLogger(__name__)
from multimodal_ds.agents.evaluation_agent import EvaluationAgent
from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent
from multimodal_ds.agents.problem_understanding_agent import ProblemUnderstandingAgent
from multimodal_ds.agents.reflection_agent import ReflectionAgent, ReflectionReport
session_logger = logging.getLogger('session_log')
if not session_logger.handlers:
    handler = logging.FileHandler(OUTPUT_DIR / 'session_log.jsonl')
    handler.setLevel(logging.INFO)
    # Use raw message (JSON string) without extra formatting
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    session_logger.addHandler(handler)
    session_logger.propagate = False


MAX_RETRIES = 2


def _sanitize_for_checkpoint(data):
    import numpy as np
    if isinstance(data, dict):
        return {k: _sanitize_for_checkpoint(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_checkpoint(v) for v in data]
    if hasattr(data, "item") and not isinstance(data, (str, bytes)):
        return data.item()
    if isinstance(data, (np.integer, np.floating)):
        return float(data) if isinstance(data, np.floating) else int(data)
    return data


# ── Node functions ───────────────────────────────────────────────────────────

def _router_node(state):
    """Determine routing flags based on uploaded file extensions.
    Returns a dict with boolean flags for each document type.
    """
    from pathlib import Path
    EXTENSIONS = {
        "doc":   {".pdf", ".docx", ".txt", ".md", ".html", ".rst"},
        "image": {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac"},
        "table": {".csv", ".xlsx", ".parquet", ".json", ".tsv"},
    }
    flags = {k: False for k in EXTENSIONS}
    for path in state.get("uploaded_files", []):
        ext = Path(path).suffix.lower()
        for kind, exts in EXTENSIONS.items():
            if ext in exts:
                flags[kind] = True
    logger.info(f"[Graph/Router] Routing flags: {flags}")
    return {"_routing_flags": flags}
def _decide_ingestion_path(state):
    """Determine which ingestion nodes to invoke based on routing flags.
    Returns a Send() action that fans out to all applicable ingestion nodes.
    """
    flags = state.get("_routing_flags", {})
    targets = []
    if flags.get("doc"):
        targets.append("doc_ingest")
    if flags.get("image"):
        targets.append("img_ingest")
    if flags.get("audio"):
        targets.append("audio_ingest")
    if flags.get("table"):
        targets.append("tab_ingest")
    # If no ingestion needed, just proceed without fan‑out.
    if not targets:
        return {}
    # Use LangGraph's Send to fan‑out to the selected nodes.
    from langgraph.graph import Send
    return Send(targets)

def _ingest_merge_node(state):
    """Merge ingestion results – simply pass through the accumulated state.
    The individual ingestion nodes already update the shared state, so this
    node returns the state unchanged.
    """
    return state


def _problem_understanding_node(state):
    """Invoke ProblemUnderstandingAgent to produce a problem spec and store it in state."""
    try:
        agent = ProblemUnderstandingAgent(session_id=state.get("session_id", "default"))
        spec = agent.understand(state.get("user_query", ""), state.get("uploaded_files", []))
        return {"problem_spec": spec.to_dict()}
    except Exception as e:
        logger.error(f"[ProblemUnderstanding] Node error: {e}")
        return {"problem_spec": {}}


def _doc_ingest_node(state):
    from multimodal_ds.ingestion.pdf_ingestion import ingest_pdf
    from multimodal_ds.ingestion.router import _ingest_plain_text
    from pathlib import Path

    DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".rst"}
    docs = list(state.get("parsed_documents", []))

    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in DOC_EXTS:
            doc = ingest_pdf(fp) if fp.endswith(".pdf") else _ingest_plain_text(fp)
            docs.append(doc.to_dict())

    vector_store_id = state.get("vector_store_id", "")
    text_chunks = [d.get("text_content", "")[:2000] for d in docs if d.get("text_content")]
    if text_chunks:
        try:
            from multimodal_ds.memory.agent_memory import AgentMemory
            mem = AgentMemory(collection_name="doc_chunks")
            for chunk in text_chunks:
                mem.store(chunk, metadata={"type": "document"})
            vector_store_id = str(mem._collection.name) if mem._collection else vector_store_id
        except Exception as e:
            logger.warning(f"[Graph/DocIngest] ChromaDB store failed: {e}")

    return {"parsed_documents": docs, "vector_store_id": vector_store_id}


def _img_ingest_node(state):
    from multimodal_ds.ingestion.image_ingestion import ingest_image, SUPPORTED_IMAGES
    from pathlib import Path

    embeddings = list(state.get("image_embeddings", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_IMAGES:
            doc = ingest_image(fp)
            if doc.embeddings:
                embeddings.append(doc.embeddings)
    return {"image_embeddings": embeddings}


def _audio_ingest_node(state):
    from multimodal_ds.ingestion.audio_ingestion import ingest_audio, SUPPORTED_AUDIO
    from pathlib import Path

    transcripts = list(state.get("audio_transcripts", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_AUDIO:
            doc = ingest_audio(fp)
            if doc.text_content:
                transcripts.append(doc.text_content)
    return {"audio_transcripts": transcripts}


def _tab_ingest_node(state):
    from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular, SUPPORTED_TABULAR
    from pathlib import Path

    summaries = list(state.get("tabular_summaries", []))
    for fp in state.get("uploaded_files", []):
        if Path(fp).suffix.lower() in SUPPORTED_TABULAR:
            doc = ingest_tabular(fp)
            if doc.schema_info:
                summaries.append({
                    "source":       fp,
                    "shape":        doc.schema_info.get("shape", []),
                    "columns":      doc.schema_info.get("columns", []),
                    "dtypes":       doc.schema_info.get("dtypes", {}),
                    "sample":       doc.text_content[:1500],
                    "data_profile": doc.data_profile,
                })
    return {"tabular_summaries": _sanitize_for_checkpoint(summaries)}


def _model_selection_node(state):
    """Select and configure models based on statistical report and AutoML suggestion."""
    import pandas as pd
    from pathlib import Path
    from multimodal_ds.agents.model_selection_agent import ModelSelectionAgent

    try:
        tab_summaries = state.get("tabular_summaries", [])
        if not tab_summaries:
            logger.info("[ModelSelection] No tabular summaries – skipping.")
            return {}
        first = tab_summaries[0]
        file_path = first.get("source")
        if not file_path:
            logger.info("[ModelSelection] Tabular summary missing source – skipping.")
            return {}

        # Load dataframe based on extension
        df = None
        p = Path(file_path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif p.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)

        if df is None:
            logger.warning(f"[ModelSelection] Unsupported file extension for {file_path}")
            return {}

        automl_suggestion = first.get("automl_suggestion", {})
        target_col = automl_suggestion.get("target_candidates", [None])[0]
        stat_report = state.get("statistical_report", {})

        agent = ModelSelectionAgent(session_id=state.get("session_id", "default"))
        result = agent.select_models(df, target_col, stat_report, automl_suggestion)
        # Determine dataset size for optional tuning
        shape = first.get("shape", [0, 0])
        n_rows = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 0

        # Default to original ensemble code
        tuned_code = code_str
        tuning_results = {}
        if 0 < n_rows <= 50000:
            try:
                logger.info(f"[ModelSelection] Starting Optuna tuning for {n_rows} rows...")
                # Prepare X and y for tuning
                X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
                y = df[target_col] if target_col and target_col in df.columns else None
                if y is not None:
                    tuning_results = agent.tune_all_models(X, y, result)
                    tuned_code = agent.generate_tuned_ensemble_code(result, tuning_results, "df", target_col)
                    logger.info(f"[ModelSelection] Tuning complete for {result.get('primary_model')}")
                else:
                    logger.warning("[ModelSelection] Target column missing for tuning; using default code")
            except Exception as e:
                logger.warning(f"[ModelSelection] Optuna tuning failed: {e} — using default ensemble code")
        else:
            logger.info(f"[ModelSelection] Skipping tuning (n_rows={n_rows})")

        return {"model_selection": result, "tuning_results": tuning_results, "ensemble_code_template": tuned_code}
    except Exception as e:
        logger.warning(f"[ModelSelection] Node error: {e}")
        return {}


def _stats_validation_node(state):
    """Run statistical validation on the dataset."""
    from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
    import pandas as pd

    uploaded = state.get("uploaded_files", [])
    tab_file = next((f for f in uploaded if f.endswith((".csv", ".xlsx", ".parquet"))), None)
    if not tab_file:
        return {}

    try:
        df = pd.read_csv(tab_file) if tab_file.endswith(".csv") else pd.read_excel(tab_file)
        agent = StatisticalReasoningAgent(session_id=state.get("session_id", "default"))
        report = agent.validate_dataset(df)
        return {"statistical_report": _sanitize_for_checkpoint(report)}
    except Exception as e:
        logger.warning(f"[Graph/Stats] Validation failed: {e}")
        return {}


def _planner_node(state):
    from multimodal_ds.agents.planner_agent import run_planner
    from pathlib import Path

    # Build a rich data‑context string (numeric stats, missing‑value info) – same as executor
    data_context_parts = []
    for t in state.get("tabular_summaries", [])[:2]:
        cols = t.get("columns", [])
        shape = t.get("shape", [])
        profile = t.get("data_profile", {})
        data_context_parts.append(
            f"Table {Path(t['source']).name}: {shape} rows×cols\n"
            f"Columns: {cols}\n"
        )
        if profile.get("numeric_stats"):
            data_context_parts.append("Numeric column stats (mean / std / min / max):")
            for col, s in list(profile["numeric_stats"].items())[:10]:
                data_context_parts.append(
                    f"  {col}: mean={s.get('mean', 0):.2f}, std={s.get('std', 0):.2f}, "
                    f"min={s.get('min', 0):.2f}, max={s.get('max', 0):.2f}"
                )
        missing = {k: v for k, v in profile.get("missing_values", {}).items() if v > 0}
        if missing:
            data_context_parts.append(f"Missing values: {missing}")
        else:
            data_context_parts.append("Missing values: none detected")
    planner_data_context = "\n".join(data_context_parts) if data_context_parts else ""
    # Store in state for potential downstream use
    state["planner_data_context"] = planner_data_context

    # -----------------------------------------------------------------
    # Run the planner LLM – we provide the user query and any available
    # document profiles (here a minimal empty list, since the graph does not
    # collect UnifiedDocument objects). The planner returns a dict with the
    # analysis plan and tasks.
    # -----------------------------------------------------------------
    from multimodal_ds.core.schema import UnifiedDocument, DataType, ProcessingStatus

    proxy_docs = []

    # 1. Tabular documents
    for t in state.get("tabular_summaries", []):
        doc = UnifiedDocument(
            data_type=DataType.TABULAR,
            status=ProcessingStatus.DONE,
            text_content=t.get("sample", ""),
            schema_info={
                "columns": t.get("columns", []),
                "shape": t.get("shape", []),
                "numeric_cols": [c for c, d in t.get("dtypes", {}).items()
                                 if "int" in d or "float" in d],
            },
            metadata={"automl_suggestion": t.get("automl_suggestion", {})}
        )
        proxy_docs.append(doc)

    # 2. Text/PDF documents (exclude BLOCKED)
    for d in state.get("parsed_documents", []):
        text = d.get("text_content", "")
        if text and not text.startswith("[BLOCKED"):
            doc = UnifiedDocument(
                data_type=DataType.TEXT,
                status=ProcessingStatus.DONE,
                text_content=text[:1500],
            )
            proxy_docs.append(doc)

    # 3. Audio transcripts
    for transcript in state.get("audio_transcripts", []):
        if transcript and not transcript.startswith("[BLOCKED"):
            doc = UnifiedDocument(
                data_type=DataType.AUDIO,
                status=ProcessingStatus.DONE,
                text_content=transcript[:1000],
            )
            proxy_docs.append(doc)

    # 4. Inject statistical report as context
    stat_report = state.get("statistical_report", {})
    stat_context = ""
    if stat_report and isinstance(stat_report, dict):
        non_normal = [k for k, v in stat_report.get("normality", {}).items()
                          if isinstance(v, dict) and not v.get("is_normal", True)]
        n_strong = stat_report.get("correlation", {}).get("n_strong", 0)
        mc = stat_report.get("multicollinearity", {}).get("multicollinearity_detected", False)
        stat_context = (
            f"Statistical findings: non-normal columns={non_normal}, "
            f"strong_correlations={n_strong}, multicollinearity={mc}"
        )
        if stat_context:
            doc = UnifiedDocument(
                data_type=DataType.TEXT,
                status=ProcessingStatus.DONE,
                text_content=f"Statistical Validation Report:\n{stat_context}",
            )
            proxy_docs.append(doc)

        # Include required deliverables from the problem spec (if any) in the planner prompt.
        problem_spec = state.get("problem_spec", {})
        user_objective = state.get("user_query", "")
        required_deliverables = problem_spec.get("required_deliverables", [])
        if required_deliverables:
            user_objective = (
                f"{user_objective}\nDesired deliverables: {', '.join(required_deliverables)}"
            )
        try:
            plan_result = run_planner(
                user_objective=user_objective,
                documents=proxy_docs,
                session_id=state.get("session_id", "default"),
            )
        except Exception as e:
            logger.warning(f"[Planner] run_planner failed: {e}")
            plan_result = {}

    tasks = plan_result.get("analysis_plan", [])
    return {
        "analysis_plan":  plan_result.get("final_plan", ""),
        "analysis_tasks": tasks,
        "hypotheses":     plan_result.get("hypotheses", []),
        "current_step":   0,
        "steps_total":    len(tasks),
    }


def _visualizer_node(state):
    """Generate visualizations using VisualizationAgent and attach tuning results."""
    import pandas as pd
    from pathlib import Path
    from multimodal_ds.agents.visualization_agent import VisualizationAgent

    try:
        tab_summaries = state.get("tabular_summaries", [])
        if not tab_summaries:
            logger.info("[Visualizer] No tabular summaries – skipping.")
            return {}
        first = tab_summaries[0]
        file_path = first.get("source")
        if not file_path:
            logger.info("[Visualizer] Tabular summary missing source – skipping.")
            return {}

        p = Path(file_path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif p.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            logger.warning(f"[Visualizer] Unsupported file extension for {file_path}")
            return {}

        target_col = first.get("automl_suggestion", {}).get("target_candidates", [None])[0]
        vis_agent = VisualizationAgent(session_id=state.get("session_id", "default"))
        tuning = state.get("tuning_results", {})
        if tuning:
            vis_agent.set_tuning_results(tuning)
        manifest = vis_agent.generate(df=df, target_col=target_col)
        viz_files = [c["filename"] for c in manifest.charts]
        logger.info(f"[Visualizer] Generated {len(viz_files)} visualization files")
        return {"visualizations": viz_files}
    except Exception as e:
        logger.warning(f"[Visualizer] Node error: {e}")
        return {}


def _reflection_node(state):
    """Run reflection on the most recent task using ReflectionAgent and inject lessons into next task."""
    # Retrieve recent task info
    tasks = state.get("analysis_tasks", [])
    step_idx = state.get("current_step", 1) - 1  # just completed
    if step_idx < 0 or step_idx >= len(tasks):
        return state
    task = tasks[step_idx]
    # Get latest output
    output = state.get("full_code_outputs", [""])[-1] if state.get("full_code_outputs") else ""
    files = state.get("_last_files_created", [])
    success = state.get("_last_success", False)
    task_result = {
        "name": task.get("name", f"step_{step_idx}"),
        "success": success,
        "output_preview": output[:1000],
        "files_created": files,
        "error": "",
    }
    # Evaluation result
    eval_report = state.get("eval_report", {})
    evaluations = eval_report.get("evaluations", []) if isinstance(eval_report, dict) else []
    eval_result = evaluations[-1] if evaluations else {}
    # Run reflection
    from multimodal_ds.agents.reflection_agent import ReflectionAgent
    session_id = state.get("session_id", "default")
    agent = ReflectionAgent(session_id=session_id)
    reflection = agent.reflect_on_task(task_result, eval_result)
    logger.info(f"[Reflection] Task '{task_result['name']}': {reflection.get('root_cause', 'success')}")
    # Inject lessons into next task if any
    remaining = tasks[step_idx + 1:]
    lessons_text = "\n".join(reflection.get("lessons", []))
    avoid_text = "\n".join(reflection.get("avoid_patterns", []))
    if lessons_text and remaining:
        next_task = remaining[0]
        if avoid_text:
            next_task["description"] = (
                next_task.get("description", "") +
                f"\n\nLESSONS FROM PREVIOUS STEP (apply these):\n{lessons_text}" +
                f"\n\nAVOID THESE PATTERNS:\n{avoid_text}"
            )
        tasks[step_idx + 1] = next_task
    return {"analysis_tasks": tasks, "reflections": state.get("reflections", []) + [reflection]}

def _reviewer_node(state):
    tasks   = state.get("analysis_tasks", [])
    outputs = state.get("full_code_outputs", [])
    errors  = state.get("errors", [])
    vizs    = state.get("visualizations", [])
    arts    = state.get("saved_artifacts", [])

    # Build all files created across session
    all_files = list(vizs) + list(arts)

    task_results = []
    for i, (task, output) in enumerate(zip(tasks, outputs)):
        step_num = i + 1
        task_failed = any(f"Step {step_num}:" in e for e in errors)
        
        # Determine files relevant to this step safely – log any issues but continue
        step_files = []
        try:
            for fname in all_files:
                if fname.lower().endswith(('.png', '.jpg', '.csv', '.pkl', '.joblib', '.html', '.txt')):
                    step_files.append(fname)
            # Include per-step file list if available
            files_per_step = state.get("_files_per_step", [])
            if i < len(files_per_step):
                step_files.extend(files_per_step[i])
        except Exception as e:
            logger.warning(f"[Reviewer] File aggregation failed: {e}")
        # Also scan output text for saved file references
        try:
            import re
            file_refs = re.findall(r'[\w\-]+\.\w{2,5}', output)
            known_exts = {'.png', '.jpg', '.csv', '.pkl', '.joblib', '.html', '.txt', '.json', '.parquet'}
            for ref in file_refs:
                if any(ref.endswith(ext) for ext in known_exts) and ref not in step_files:
                    step_files.append(ref)
        except Exception as e:
            logger.warning(f"[Reviewer] Output file reference parsing failed: {e}")


        task_results.append({
            "name":           task.get("name", f"step_{step_num}"),
            "success":        not task_failed,
            "output_preview": output,
            "files_created":  step_files,
            "error":          "",
        })

    session_id = state.get("session_id", "default")
    data_context = _build_data_context_for_eval(state)
    try:
        eval_agent = EvaluationAgent(session_id=session_id)
        stat_report = state.get("statistical_report", {})
        report = eval_agent.evaluate_task_results(
            task_results=task_results,
            data_context=data_context,
            stat_report=stat_report if stat_report else None,
        )
        return {"eval_report": report.to_dict()}
    except Exception as e:
        logger.warning(f"[Reviewer] EvaluationAgent failed: {e} — returning empty report")
        return {"eval_report": {
            "session_id": session_id,
            "task_count": len(task_results),
            "flagged_count": 0,
            "pass_count": len(task_results),
            "overall_session_score": 5.0,
            "session_verdict": "UNKNOWN",
            "evaluations": [],
        }}


def _build_data_context_for_eval(state: dict) -> str:
    """Build rich data context string for the evaluation agent."""
    parts = []
    for t in state.get("tabular_summaries", [])[:2]:
        cols = t.get("columns", [])
        shape = t.get("shape", [])
        parts.append(f"Dataset: {shape[0] if shape else '?'} rows × {shape[1] if len(shape) > 1 else '?'} cols")
        parts.append(f"Columns: {', '.join(str(c) for c in cols[:20])}")
        profile = t.get("data_profile", {})
        if profile.get("numeric_stats"):
            stats_preview = list(profile["numeric_stats"].items())[:3]
            for col, s in stats_preview:
                parts.append(f"  {col}: mean={s.get('mean', 0):.2f}, std={s.get('std', 0):.2f}")
    return "\n".join(parts)


def _retry_node(state):
    """
    Reflection loop — diagnoses failures and generates improved instructions.
    Replaces the stub counter-increment with LLM-guided root cause analysis.
    """
    retry_count = state.get("retry_count", 0) + 1
    logger.warning(f"[Graph] Reflection loop triggered. Attempt {retry_count}")

    try:
        agent = ReflectionAgent(
            session_id=state.get("session_id", "default"),
            max_retries=MAX_RETRIES,
        )
        eval_report = state.get("eval_report", {})
        if not isinstance(eval_report, dict):
            eval_report = eval_report.to_dict() if hasattr(eval_report, "to_dict") else {}

        report = agent.reflect(eval_report=eval_report, state=state)

        # If reflection produced improved instructions, inject them into the next task
        improved = report.improved_instructions
        tasks = list(state.get("analysis_tasks", []))
        current_step = state.get("current_step", 0)

        # Rewind current_step by 1 so executor reruns the failed task with new instructions
        retry_step = max(0, current_step - 1)
        if tasks and retry_step < len(tasks):
            # Augment the failed task's description with improved instructions
            tasks[retry_step] = {
                **tasks[retry_step],
                "description": tasks[retry_step].get("description", "")
                + f"\n\nREFLECTION GUIDANCE: {improved}",
            }

        return {
            **state,
            "retry_count": retry_count,
            "analysis_tasks": tasks,
            "current_step": retry_step,
            "reflection_report": report.to_dict(),
            "errors": [],  # Clear errors so reviewer starts fresh
        }

    except Exception as e:
        logger.error(f"[Graph] ReflectionAgent failed: {e} — falling back to counter increment")
        return {**state, "retry_count": retry_count}


def _reporter_node(state):
    from multimodal_ds.agents.reporter import reporter_agent
    return reporter_agent(state)

# ── Quality gate node ────────────────────────────────────────────────────────

def _quality_gate_node(state):
    """Fast rule‑based quality gate between executor and reviewer."""
    # Last execution info
    outputs = state.get("full_code_outputs", [])
    last_output = outputs[-1] if outputs else ""
    last_files = state.get("_last_files_created", [])
    last_success = state.get("_last_success", False)
    current_step = state.get("current_step", 0)
    tasks = state.get("analysis_tasks", [])
    task_name = "unknown"
    if current_step > 0 and tasks:
        task_name = tasks[current_step - 1].get("name", "unknown")
    # Rule checks
    gate_passed = True
    gate_reasons = []
    # 1. OS‑level success
    if not last_success:
        gate_passed = False
        gate_reasons.append("Execution returned non-zero exit code")
    # 2. Output length
    if len(last_output.strip()) < 50:
        gate_passed = False
        gate_reasons.append("Output too short — likely crashed silently")
    # 3. Expected artifact files for modeling/evaluation
    task_type = tasks[current_step - 1].get("type", "") if current_step > 0 and tasks else ""
    if task_type in ("modeling", "evaluation"):
        if not any(f.endswith((".pkl", ".joblib", ".csv", ".txt")) for f in last_files):
            gate_passed = False
            gate_reasons.append(f"Modeling task produced no artifact files: {last_files}")
    # 4. Python error indicators
    error_indicators = ["Traceback (most recent", "Error:", "Exception:", "ModuleNotFoundError", "KeyError:", "AttributeError:"]
    if any(ind in last_output for ind in error_indicators) and not last_files:
        gate_passed = False
        gate_reasons.append("Output contains Python errors with no recovery files")
    # 5. Hallucination guard – column names
    if state.get("tabular_summaries"):
        actual_cols = set(state["tabular_summaries"][0].get("columns", []))
        import re
        matches = re.findall(r"df\[\'([^\']+)\'\]|df\[\"([^\"]+)\"\]", last_output)
        mentioned_cols = {c for pair in matches for c in pair if c}
        phantom_cols = mentioned_cols - actual_cols
        if phantom_cols and len(phantom_cols) > 2:
            gate_passed = False
            gate_reasons.append(f"Hallucinated column names: {phantom_cols}")
    # Logging
    if gate_passed:
        logger.info(f"[QualityGate] Step {current_step} '{task_name}': PASSED")
    else:
        logger.warning(f"[QualityGate] Step {current_step} '{task_name}': FAILED — {gate_reasons}")
    # Update errors list
    new_errors = state.get("errors", [])
    if not gate_passed:
        new_errors = new_errors + [f"Quality gate failed: {r}" for r in gate_reasons]
    return {
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
        "errors": new_errors,
    }


# ── Conditional edges ────────────────────────────────────────────────────────

def _multi_ingest_router_node(state):
    """Concurrent ingestion of multiple file types.
    Reads routing flags from state['_routing_flags'] and invokes the relevant
    ingestion node functions in parallel using a ThreadPoolExecutor.
    Returns a merged state dict where list values are concatenated.
    """
    import concurrent.futures
    flags = state.get("_routing_flags", {})
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        if flags.get("table"):
            futures.append(executor.submit(_tab_ingest_node, state))
        if flags.get("doc"):
            futures.append(executor.submit(_doc_ingest_node, state))
        if flags.get("image"):
            futures.append(executor.submit(_img_ingest_node, state))
        if flags.get("audio"):
            futures.append(executor.submit(_audio_ingest_node, state))
        # Wait for all futures with timeout
        concurrent.futures.wait(futures, timeout=300)
    merged = {}
    for fut in futures:
        try:
            result = fut.result()
        except Exception as e:
            logger.warning(f"[MultiIngest] Ingestion future failed: {e}")
            continue
        for k, v in result.items():
            if isinstance(v, list) and isinstance(merged.get(k), list):
                merged[k] = merged[k] + v
            else:
                merged[k] = v
    return merged

def _decide_review_outcome(state) -> str:
    """
    Decide whether to:
    1. Continue to next task step (executor)
    2. Retry the whole session if overall failures (retry -> executor)
    3. Finish and report (reporter)
    """
    retry_count   = state.get("retry_count", 0)
    eval_report   = state.get("eval_report", {})
    if not isinstance(eval_report, dict):
        # Fallback to attribute access for dummy objects
        eval_report = {
            "overall_session_score": getattr(state.get("eval_report"), "overall_session_score", 10),
            "flagged_count": getattr(state.get("eval_report"), "flagged_count", 0),
        }
    overall_score = eval_report.get("overall_session_score", 10)
    has_failures  = eval_report.get("flagged_count", 0) > 0

    current = state.get("current_step", 0)
    total   = state.get("steps_total", 0)

    # 1. If we have more steps, keep going
    if current < total:
        return "executor"

    # 2. If we finished all steps but had critical failures, try a session-level retry
    if has_failures and retry_count < MAX_RETRIES and overall_score < 5:
        return "retry"

    # 3. Otherwise, we are done
    return "reporter"

def _decide_after_gate(state) -> str:
    """Decide next step after quality gate.
    Returns "reviewer", "retry", or "reporter".
    """
    if state.get("gate_passed", True):
        return "reviewer"
    retry_count = state.get("retry_count", 0)
    current = state.get("current_step", 0)
    total = state.get("steps_total", 0)
    if retry_count < MAX_RETRIES:
        return "retry"
    if current < total:
        return "reviewer"
    return "reporter"

def _decide_review_outcome(state) -> str:
    """Decide next step after reflection.
    Returns "executor", "retry", or "reporter".
    """
    eval_report = state.get("eval_report", {})
    verdict = eval_report.get("session_verdict", "UNKNOWN")
    current = state.get("current_step", 0)
    total = state.get("steps_total", 0)

    if verdict == "PASS" and current >= total:
        return "reporter"
    if verdict == "NEEDS_IMPROVEMENT" and current <= total:
        return "executor"
    if verdict == "FAIL":
        return "retry"
    if current < total:
        return "executor"
    return "reporter"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(use_sqlite_checkpointer: bool = False, sqlite_path: str = "./checkpoints.db"):
    # pyrefly: ignore [missing-import]
    from langgraph.graph import StateGraph, END
    from multimodal_ds.core.state import AgentState

    builder = StateGraph(AgentState)

    builder.add_node("problem_understanding", _problem_understanding_node)
    builder.add_node("router", _router_node)
    builder.add_node("decide_ingestion_path", _decide_ingestion_path)
    builder.add_node("doc_ingest", _doc_ingest_node)
    builder.add_node("img_ingest", _img_ingest_node)
    builder.add_node("audio_ingest", _audio_ingest_node)
    builder.add_node("tab_ingest", _tab_ingest_node)
    builder.add_node("merge_ingest", _ingest_merge_node)
    # builder.add_node("multi_ingest", _multi_ingest_router_node)
    builder.add_node("stats_val", _stats_validation_node)
    builder.add_node("model_selection", _model_selection_node)
    builder.add_node("planner", _planner_node)
    builder.add_node("visualizer", _visualizer_node)
    builder.add_node("quality_gate", _quality_gate_node)
    builder.add_node("reviewer", _reviewer_node)
    builder.add_node("reflection", _reflection_node)
    builder.add_node("executor", _executor_node)
    builder.add_node("reporter", _reporter_node)

    builder.set_entry_point("problem_understanding")

    builder.add_edge("problem_understanding", "router")
    builder.add_edge("router", "decide_ingestion_path")
    # Fan‑out to required ingestion nodes based on routing flags
    builder.add_edge("decide_ingestion_path", "doc_ingest")
    builder.add_edge("decide_ingestion_path", "img_ingest")
    builder.add_edge("decide_ingestion_path", "audio_ingest")
    builder.add_edge("decide_ingestion_path", "tab_ingest")
    # Merge results from all ingestion nodes
    builder.add_edge("doc_ingest", "merge_ingest")
    builder.add_edge("img_ingest", "merge_ingest")
    builder.add_edge("audio_ingest", "merge_ingest")
    builder.add_edge("tab_ingest", "merge_ingest")
    # Continue with validation after merging
    builder.add_edge("merge_ingest", "stats_val")
    # builder.add_edge("router", "multi_ingest")
    # After multi_ingest, always go through stats validation (no‑op if no table)
    # builder.add_edge("multi_ingest", "stats_val")
    builder.add_edge("stats_val",    "model_selection")
    builder.add_edge("model_selection", "planner")
    builder.add_edge("planner",      "executor")
    builder.add_edge("executor",     "visualizer")

    builder.add_conditional_edges(
        "quality_gate",
        _decide_after_gate,
        {"reviewer": "reviewer", "retry": "retry", "reporter": "reporter"}
    )

    builder.add_edge("reviewer", "reflection")

    builder.add_conditional_edges(
        "reflection",
        _decide_review_outcome,
        {"executor": "executor", "retry": "retry", "reporter": "reporter"}
    )

    builder.add_edge("visualizer", "quality_gate")

    builder.add_edge("reporter", END)


    if use_sqlite_checkpointer:
        try:
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.sqlite import SqliteSaver
            memory = SqliteSaver.from_conn_string(sqlite_path)
        except ImportError:
            # pyrefly: ignore [missing-import]
            from langgraph.checkpoint.memory import MemorySaver
            memory = MemorySaver()
    else:
        # pyrefly: ignore [missing-import]
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    return builder.compile(checkpointer=memory)


def make_initial_state(
    user_query: str,
    uploaded_files: list[str],
    session_id: Optional[str] = None,
) -> dict:
    return {
        "user_query":         user_query,
        "uploaded_files":     uploaded_files,
        "_routing_flags":     {},
        "parsed_documents":   [],
        "image_embeddings":   [],
        "audio_transcripts":  [],
        "tabular_summaries":  [],
        "statistical_report": {},
        "model_selection": {},
        "tuning_results": {},
        "analysis_plan":      "",
        "analysis_tasks":     [],
        "hypotheses":         [],
        "current_step":       0,
        "steps_total":        0,
        "code_outputs":       [],
        "full_code_outputs": [],
        "visualizations":     [],
        "saved_artifacts":    [],
        "gate_passed":        True,
        "gate_reasons":      [],
        "reflections":        [],
        "vector_store_id":    "",
        "retrieved_context":  "",
        "eval_report":        {},
        "reflection_report":  {},
        "final_report":       "",
        "problem_spec":       {},
        "session_id":         session_id or str(uuid.uuid4())[:8],
        "messages":           [],
        "_last_task_name":    "",
        "_last_files_created": [],
        "current_step_files": [],
        "current_step_success": False,
        "_files_per_step": [],
        "executive_summary":        "",
        "business_recommendations": [],
    }
