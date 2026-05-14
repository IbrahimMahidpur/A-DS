"""
Visualization Agent — generates a production-grade Plotly chart gallery.

Chart suite (auto-selected based on data shape):
  1. data_quality        — missing value bar chart (always generated)
  2. distributions       — histogram grid for all numeric columns
  3. correlation_heatmap — Pearson correlation matrix (≥2 numeric cols)
  4. target_analysis     — class balance + box plots (binary/categorical target)
  5. scatter_matrix      — pair plot coloured by target (≥50 rows)
  6. feature_importance  — bar chart from feature_importance.csv if present
  7. roc_curve           — Logistic Regression baseline ROC (binary target, sklearn)

Each chart:
  - Saved as .html (self-contained Plotly interactive file)
  - Gets an LLM-generated narrative paragraph via Ollama
  - Is registered in ChartManifest with type, filename, title, narrative, data_shape

Message bus integration:
  - Publishes VIZ_REQUEST  at the start of generate()
  - Publishes VIZ_COMPLETE at the end with chart_count in payload

Graceful degradation:
  - _PLOTLY_AVAILABLE flag — if plotly isn't installed, generate() returns
    an empty manifest without raising.
  - All individual chart methods are wrapped in try/except so one failing
    chart never aborts the entire gallery.
  - Ollama narrative fallback — if LLM is unreachable, a rule-based string
    is used instead.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from multimodal_ds.config import OUTPUT_DIR, REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT

logger = logging.getLogger(__name__)

# ── Plotly availability flag ───────────────────────────────────────────────
# Patched to False in tests that verify graceful degradation.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("[VizAgent] plotly not installed — visualization disabled")


# ══════════════════════════════════════════════════════════════════════════
#  ChartManifest
# ══════════════════════════════════════════════════════════════════════════

class ChartManifest:
    """
    Registry of all charts generated in a session.

    Charts are stored as plain dicts so the manifest is trivially JSON-serialisable
    and survives LangGraph checkpoint serialisation.

    Schema per chart entry:
        {
            "chart_type":  str,        # e.g. "correlation_heatmap"
            "filename":    str,        # e.g. "correlation_heatmap.html"
            "title":       str,
            "narrative":   str,        # LLM or rule-based insight text
            "data_shape":  [int, int], # [rows, cols] at time of generation
        }
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.charts: List[Dict[str, Any]] = []

    def add(
        self,
        chart_type: str,
        filename: str,
        title: str,
        narrative: str,
        data_shape: tuple,
    ) -> None:
        self.charts.append({
            "chart_type": chart_type,
            "filename":   filename,
            "title":      title,
            "narrative":  narrative,
            "data_shape": list(data_shape),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "chart_count": len(self.charts),
            "charts":      self.charts,
        }

    def save(self, output_dir: Path) -> Path:
        """Persist manifest as chart_manifest.json in output_dir."""
        path = Path(output_dir) / "chart_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ══════════════════════════════════════════════════════════════════════════
#  VisualizationAgent
# ══════════════════════════════════════════════════════════════════════════

class VisualizationAgent:
    """
    Generates a standard Plotly chart gallery for any tabular dataset.

    Usage:
        agent = VisualizationAgent(session_id="abc123")
        manifest = agent.generate(df=df, target_col="churn")
        print(manifest.to_dict())
    """

    def __init__(self, session_id: str, working_dir: Optional[str] = None):
        self.session_id = session_id
        # working_dir is the BASE directory — session subdir appended here
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def _publish_viz_request(self) -> None:
        """Publish VIZ_REQUEST to signal that visualization has started."""
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
            bus = get_bus()
            bus.publish(AgentMessage(
                msg_type=MessageType.VIZ_REQUEST,
                payload={"session_id": self.session_id},
                sender="visualization_agent",
                session_id=self.session_id,
            ))
        except Exception as e:
            logger.debug(f"[VizAgent] Bus publish VIZ_REQUEST failed: {e}")


    # ── Main entry point ───────────────────────────────────────────────────

    def generate(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> ChartManifest:
        """
        Generate the full chart gallery and return a ChartManifest.

        Steps:
          1. Publish VIZ_REQUEST to bus
          2. Generate each chart (failures are isolated per chart)
          3. Save manifest JSON
          4. Publish VIZ_COMPLETE to bus
          5. Return manifest
        """
        manifest = ChartManifest(session_id=self.session_id)

        # ── Bus: VIZ_REQUEST ───────────────────────────────────────────────
        self._publish_viz_request()

        if not _PLOTLY_AVAILABLE:
            logger.warning("[VizAgent] plotly not available — returning empty manifest")
            self._publish_viz_complete(manifest)
            return manifest

        if df is None or df.empty:
            logger.warning("[VizAgent] Empty dataframe — returning empty manifest")
            self._publish_viz_complete(manifest)
            return manifest

        logger.info(
            f"[VizAgent] Generating chart gallery for session {self.session_id} "
            f"— df shape {df.shape}, target={target_col!r}"
        )

        numeric_cols = list(df.select_dtypes(include=["number"]).columns)

        # ── Chart 1: Data quality / missing values (always) ────────────────
        self._chart_missing_values(df, manifest)

        # ── Chart 2: Feature distributions ────────────────────────────────
        if numeric_cols:
            self._chart_distributions(df, numeric_cols, target_col, manifest)

        # ── Chart 3: Correlation heatmap ───────────────────────────────────
        if len(numeric_cols) >= 2:
            self._chart_correlation_heatmap(df, numeric_cols, manifest)

        # ── Chart 4: Target analysis ───────────────────────────────────────
        if target_col and target_col in df.columns:
            self._chart_target_analysis(df, target_col, numeric_cols, manifest)

        # ── Chart 5: Scatter matrix (larger datasets only) ─────────────────
        if len(df) >= 50 and len(numeric_cols) >= 2:
            self._chart_scatter_matrix(df, numeric_cols, target_col, manifest)

        # ── Chart 6: Feature importance (if artifact exists) ───────────────
        fi = self._find_feature_importance()
        if fi:
            self._chart_feature_importance(fi, manifest)

        # ── Chart 7: ROC curve (binary target, sklearn required) ───────────
        if target_col and target_col in df.columns:
            self._chart_roc_curve(df, target_col, numeric_cols, manifest)

        # Optuna results chart (if tuning was run)
        tuning_results = getattr(self, "_tuning_results", None)
        if tuning_results and tuning_results.get("best_overall_model"):
            self._chart_optuna_results(tuning_results, manifest)

        # Save manifest JSON
        manifest.save(self.working_dir)
        self._publish_viz_complete(manifest)

        logger.info(
            f"[VizAgent] Gallery complete — {len(manifest.charts)} charts "
            f"saved to {self.working_dir}"
        )
        return manifest

    # ── Individual chart generators ────────────────────────────────────────

    def _chart_missing_values(self, df: pd.DataFrame, manifest: ChartManifest) -> None:
        """Bar chart of missing value counts per column."""
        try:
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing"]

            fig = px.bar(
                missing,
                x="Column",
                y="Missing",
                title="Missing Values per Column",
                color="Missing",
                color_continuous_scale="Reds",
            )
            fig.update_layout(showlegend=False)

            filename = "data_quality.html"
            self._save_chart(fig, filename)

            narrative = self._get_narrative(
                "data quality showing missing values per column",
                df,
                fallback=(
                    f"The dataset has {df.shape[0]} rows × {df.shape[1]} columns. "
                    f"Total missing cells: {int(df.isnull().sum().sum())}. "
                    f"Columns with highest missingness should be imputed or dropped before modelling."
                ),
            )
            manifest.add("data_quality", filename, "Data Quality Overview", narrative, df.shape)

        except Exception as e:
            logger.error(f"[VizAgent] data_quality chart failed: {e}")

    def _chart_distributions(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
        manifest: ChartManifest,
    ) -> None:
        """Histogram grid for all numeric columns."""
        try:
            cols_to_plot = [c for c in numeric_cols if c != target_col][:12]
            if not cols_to_plot:
                return

            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

            from plotly.subplots import make_subplots
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

            for idx, col in enumerate(cols_to_plot):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                    row=row, col=col_pos,
                )

            fig.update_layout(title_text="Feature Distributions", height=max(400, n_rows * 250))

            filename = "distributions.html"
            self._save_chart(fig, filename)

            narrative = self._get_narrative(
                "feature distributions showing the spread and skewness of numeric columns",
                df,
                fallback=(
                    f"Histograms of {len(cols_to_plot)} numeric features reveal the distributional "
                    f"characteristics of the dataset. Skewed distributions may benefit from "
                    f"log or Box-Cox transformation before modelling."
                ),
            )
            manifest.add("distributions", filename, "Feature Distributions", narrative, df.shape)

        except Exception as e:
            logger.error(f"[VizAgent] distributions chart failed: {e}")

    def _chart_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        manifest: ChartManifest,
    ) -> None:
        """Pearson correlation heatmap — skipped if fewer than 2 numeric columns."""
        try:
            if len(numeric_cols) < 2:
                logger.debug("[VizAgent] Skipping correlation heatmap — need ≥2 numeric cols")
                return

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Pearson Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            fig.update_layout(height=500)

            filename = "correlation_heatmap.html"
            self._save_chart(fig, filename)

            # Identify strong pairs for the narrative fallback
            strong = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    v = corr.iloc[i, j]
                    if abs(v) > 0.7:
                        strong.append(f"{corr.columns[i]} ↔ {corr.columns[j]} ({v:.2f})")

            fallback = (
                f"Correlation matrix of {len(numeric_cols)} numeric features. "
                + (
                    f"Strong correlations detected: {', '.join(strong[:3])}."
                    if strong
                    else "No strong correlations (|r| > 0.7) detected."
                )
            )
            narrative = self._get_narrative(
                "Pearson correlation heatmap of numeric features",
                df,
                fallback=fallback,
            )
            manifest.add(
                "correlation_heatmap", filename,
                "Feature Correlations", narrative, df.shape,
            )

        except Exception as e:
            logger.error(f"[VizAgent] correlation_heatmap chart failed: {e}")

    def _chart_target_analysis(
        self,
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: List[str],
        manifest: ChartManifest,
    ) -> None:
        """Class balance histogram + box plots of features vs target."""
        try:
            n_unique = df[target_col].nunique()
            is_binary = n_unique == 2

            # Class balance
            fig = px.histogram(
                df,
                x=target_col,
                title=f"Target Distribution — {target_col}",
                color=target_col if n_unique <= 10 else None,
            )
            filename = "target_analysis.html"
            self._save_chart(fig, filename)

            counts = df[target_col].value_counts().to_dict()
            narrative = self._get_narrative(
                f"target variable distribution for '{target_col}'",
                df,
                fallback=(
                    f"Target '{target_col}' has {n_unique} unique values. "
                    f"Class distribution: {counts}. "
                    + ("Dataset is binary classification." if is_binary else "")
                ),
            )
            manifest.add("target_analysis", filename, f"Target: {target_col}", narrative, df.shape)

            # Box plots for top numeric features vs target (binary/low-cardinality only)
            if n_unique <= 10:
                feat_cols = [c for c in numeric_cols if c != target_col][:3]
                for feat in feat_cols:
                    try:
                        fig = px.box(
                            df,
                            x=target_col,
                            y=feat,
                            title=f"{feat} by {target_col}",
                            color=target_col,
                        )
                        fname = f"{feat}_vs_target.html"
                        self._save_chart(fig, fname)
                        narrative = self._get_narrative(
                            f"box plot of '{feat}' grouped by target '{target_col}'",
                            df,
                            fallback=(
                                f"Box plot shows the distribution of '{feat}' "
                                f"across {n_unique} target classes."
                            ),
                        )
                        manifest.add(
                            "target_analysis", fname,
                            f"{feat} by {target_col}", narrative, df.shape,
                        )
                    except Exception as box_e:
                        logger.debug(f"[VizAgent] box plot {feat} failed: {box_e}")

        except Exception as e:
            logger.error(f"[VizAgent] target_analysis chart failed: {e}")

    def _chart_scatter_matrix(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
        manifest: ChartManifest,
    ) -> None:
        """Scatter matrix (pair plot) coloured by target — 50+ row datasets only."""
        try:
            cols = [c for c in numeric_cols if c != target_col][:5]
            if len(cols) < 2:
                return

            color_col = target_col if target_col and target_col in df.columns else None

            fig = px.scatter_matrix(
                df,
                dimensions=cols,
                color=color_col,
                title="Scatter Matrix",
                opacity=0.6,
            )
            fig.update_traces(diagonal_visible=False)

            filename = "scatter_matrix.html"
            self._save_chart(fig, filename)

            narrative = self._get_narrative(
                "scatter matrix showing pairwise feature relationships",
                df,
                fallback=(
                    f"Pairwise scatter plots of {len(cols)} features"
                    + (f" coloured by '{target_col}'" if color_col else "")
                    + ". Diagonal patterns suggest linear separability."
                ),
            )
            manifest.add("scatter_matrix", filename, "Scatter Matrix", narrative, df.shape)

        except Exception as e:
            logger.error(f"[VizAgent] scatter_matrix chart failed: {e}")

    def _chart_feature_importance(
        self,
        fi: Dict[str, float],
        manifest: ChartManifest,
    ) -> None:
        """Bar chart of feature importances loaded from feature_importance.csv."""
        try:
            sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])

            fig = px.bar(
                x=list(sorted_fi.values()),
                y=list(sorted_fi.keys()),
                orientation="h",
                title="Feature Importance",
                labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))

            filename = "feature_importance.html"
            self._save_chart(fig, filename)

            top = list(sorted_fi.keys())[:3]
            narrative = (
                f"Feature importance scores from the trained model. "
                f"Top features: {', '.join(top)}. "
                f"Focus model interpretation on these high-impact variables."
            )
            manifest.add(
                "feature_importance", filename,
                "Feature Importance", narrative, (len(fi), 1),
            )

        except Exception as e:
            logger.error(f"[VizAgent] feature_importance chart failed: {e}")

    def _chart_roc_curve(
        self,
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: List[str],
        manifest: ChartManifest,
    ) -> None:
        """Logistic Regression baseline ROC curve — binary target only."""
        try:
            import sklearn  # noqa: F401 — skip if not installed
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, roc_curve
            from sklearn.preprocessing import StandardScaler

            feat_cols = [c for c in numeric_cols if c != target_col]
            if not feat_cols:
                return

            sub = df[feat_cols + [target_col]].dropna()
            if sub.empty or sub[target_col].nunique() != 2:
                return

            X = sub[feat_cols].values
            y = sub[target_col].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_scaled, y)
            proba = clf.predict_proba(X_scaled)[:, 1]

            fpr, tpr, _ = roc_curve(y, proba)
            auc = roc_auc_score(y, proba)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="grey"),
                name="Random",
            ))
            fig.update_layout(
                title=f"ROC Curve — Logistic Regression Baseline (AUC={auc:.3f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )

            filename = "roc_curve.html"
            self._save_chart(fig, filename)

            narrative = (
                f"Logistic Regression baseline ROC curve for target '{target_col}'. "
                f"AUC = {auc:.3f} — "
                + (
                    "strong discriminative power."
                    if auc >= 0.8
                    else "moderate discriminative power — consider non-linear models."
                    if auc >= 0.6
                    else "weak discriminative power — further feature engineering needed."
                )
            )
            manifest.add("roc_curve", filename, "ROC Curve (Baseline)", narrative, df.shape)

        except ImportError:
            logger.debug("[VizAgent] sklearn not installed — skipping ROC curve")
        except Exception as e:
            logger.error(f"[VizAgent] roc_curve chart failed: {e}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _save_chart(self, fig: Any, filename: str) -> Path:
        """Save a Plotly figure as a self-contained HTML file."""
        path = self.working_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        logger.debug(f"[VizAgent] Saved chart → {path}")
        return path

    def _find_feature_importance(self) -> Dict[str, float]:
        """
        Look for a feature_importance.csv in the session working directory.
        Expected columns: feature, importance
        Returns empty dict if not found or malformed.
        """
        fi_path = self.working_dir / "feature_importance.csv"
        if not fi_path.exists():
            return {}
        try:
            fi_df = pd.read_csv(fi_path)
            if "feature" not in fi_df.columns or "importance" not in fi_df.columns:
                return {}
            return dict(zip(fi_df["feature"], fi_df["importance"].astype(float)))
        except Exception as e:
            logger.debug(f"[VizAgent] Could not load feature_importance.csv: {e}")
            return {}

    def _get_narrative(
        self,
        chart_desc: str,
        df: pd.DataFrame,
        fallback: str = "",
    ) -> str:
        """
        Call Ollama to generate a 2-3 sentence statistical insight for a chart.
        Falls back to the provided fallback string if the LLM is unreachable.
        """
        import httpx

        cols  = list(df.columns)
        stats = df.describe().to_string()[:400]

        prompt = (
            f"You are a data analyst. Write a 2-3 sentence statistical insight for a "
            f"{chart_desc}.\n"
            f"Data columns: {cols}\n"
            f"Summary stats:\n{stats}\n\n"
            f"Be specific and mention potential trends or patterns. "
            f"Do NOT use markdown formatting."
        )

        try:
            model = REVIEWER_MODEL.replace("ollama/", "")
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"num_predict": 150, "temperature": 0.2},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                if content:
                    return content
        except Exception as e:
            logger.debug(f"[VizAgent] Narrative LLM call failed: {e}")

        # Use fallback if LLM unavailable or returned empty
        if fallback:
            return fallback
        return f"This chart shows {chart_desc} for the provided dataset."

    # ── Message bus integration ────────────────────────────────────────────

    def _chart_optuna_results(self, tuning_results: dict, manifest: ChartManifest) -> None:
        """Generate Optuna optimization history and parameter importance charts."""
        try:
            import optuna
            import optuna.importance as oim

            best_model = tuning_results.get("best_overall_model")
            best_score = tuning_results.get("best_overall_score")

            # ------- Optimization History -------
            fig_hist = go.Figure()
            max_trial = 0
            for model_name, result in tuning_results.items():
                if model_name in {"best_overall_model", "best_overall_score"}:
                    continue
                study = result.get("study")
                if not study:
                    continue
                trials = sorted(study.trials, key=lambda t: t.number)
                xs = [t.number for t in trials]
                ys = [t.value for t in trials]
                if not xs:
                    continue
                max_trial = max(max_trial, max(xs))
                fig_hist.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=model_name))

            # Horizontal dashed line for best overall score
            if max_trial > 0 and best_score is not None:
                fig_hist.add_hline(y=best_score, line_dash="dash", annotation_text=f"Best Score: {best_score:.4f}", annotation_position="top left")

            fig_hist.update_layout(
                title="Optuna Hyperparameter Optimization History",
                xaxis_title="Trial",
                yaxis_title="Score",
                legend_title="Model",
            )
            hist_file = "optuna_history.html"
            self._save_chart(fig_hist, hist_file)
            narrative_hist = self._get_narrative(
                "Optuna optimization history for hyperparameter tuning",
                None,
                fallback="Optimization history shows how scores evolve across trials for each model.",
            )
            manifest.add("optuna_history", hist_file, "Optuna Optimization History", narrative_hist, (0, 0))

            # ------- Parameter Importance (best model) -------
            if best_model:
                best_res = tuning_results.get(best_model, {})
                study = best_res.get("study")
                if study:
                    try:
                        importances = oim.get_param_importances(study)
                    except Exception:
                        importances = {}
                    if importances:
                        fig_imp = go.Figure()
                        sorted_imp = sorted(importances.items(), key=lambda kv: kv[1])
                        params, vals = zip(*sorted_imp)
                        fig_imp.add_trace(go.Bar(x=vals, y=params, orientation='h'))
                        fig_imp.update_layout(
                            title=f"Parameter Importance — {best_model}",
                            xaxis_title="Importance",
                            yaxis_title="Parameter",
                        )
                        imp_file = "optuna_param_importance.html"
                        self._save_chart(fig_imp, imp_file)
                        narrative_imp = self._get_narrative(
                            f"Optuna parameter importance for the best model {best_model}",
                            None,
                            fallback=f"Parameter importance chart highlights which hyperparameters most affect performance for {best_model}.",
                        )
                        manifest.add("optuna_param_importance", imp_file, f"Parameter Importance — {best_model}", narrative_imp, (0, 0))
        except Exception as e:
            logger.error(f"[VizAgent] Optuna result charts failed: {e}")
        """Store tuning results for later chart generation."""
        self._tuning_results = tuning_results

        """Publish VIZ_REQUEST to signal that visualization has started."""
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
            bus = get_bus()
            bus.publish(AgentMessage(
                msg_type=MessageType.VIZ_REQUEST,
                payload={"session_id": self.session_id},
                sender="visualization_agent",
                session_id=self.session_id,
            ))
        except Exception as e:
            logger.debug(f"[VizAgent] Bus publish VIZ_REQUEST failed: {e}")

    def _publish_viz_complete(self, manifest: ChartManifest) -> None:
        """Publish VIZ_COMPLETE with chart count after gallery is built."""
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
            bus = get_bus()
            bus.publish(AgentMessage(
                msg_type=MessageType.VIZ_COMPLETE,
                payload={
                    "session_id":  self.session_id,
                    "chart_count": len(manifest.charts),
                    "charts":      [c["filename"] for c in manifest.charts],
                },
                sender="visualization_agent",
                session_id=self.session_id,
            ))
        except Exception as e:
            logger.debug(f"[VizAgent] Bus publish VIZ_COMPLETE failed: {e}")
