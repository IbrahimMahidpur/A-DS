"""
Statistical Reasoning Agent — validates statistical assumptions.
Specialist agent #2: normality, stationarity, multicollinearity, etc.
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd

from multimodal_ds.config import REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT
from multimodal_ds.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class StatisticalReasoningAgent:
    """
    Validates statistical assumptions before modeling:
    - Normality (Shapiro-Wilk, D'Agostino)
    - Stationarity (ADF test for time series)
    - Multicollinearity (VIF)
    - Homoscedasticity (Levene's test)
    - Correlation analysis
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory = AgentMemory()

    def validate_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        report = {
            "normality":          self._check_normality(df),
            "correlation":        self._check_correlation(df),
            "multicollinearity":  self._check_multicollinearity(df, target_col),
            "stationarity":       self._check_stationarity(df),
            "recommendations":    [],
        }
        report["interpretation"]  = self._interpret_findings(report, df.shape)
        report["recommendations"] = self._generate_recommendations(report)

        self.memory.store_analysis_step(
            step_name="statistical_validation",
            result=str(report["interpretation"])[:500],
            session_id=self.session_id
        )
        # Store findings in shared context pool
        from multimodal_ds.core.context_pool import get_context_pool
        pool = get_context_pool(self.session_id)
        pool.set("stat_normality", report["normality"], agent="statistical_agent")
        pool.set("stat_correlation", report["correlation"], agent="statistical_agent")
        pool.set("stat_multicollinearity", report["multicollinearity"], agent="statistical_agent")
        pool.set("stat_recommendations", report["recommendations"], agent="statistical_agent")

        return report

    def _check_normality(self, df: pd.DataFrame) -> dict:
        """Check normality for numeric columns.
        Uses Shapiro-Wilk for <=5000 samples, D'Agostino's K^2 test for larger.
        Falls back to skew/kurtosis heuristic if scipy.stats unavailable.
        Skips columns with fewer than 8 non‑null values.
        Returns a dict mapping column name -> {"is_normal": bool, "p_value": float|None, "test": str}.
        """
        result = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 8:
                continue
            try:
                from scipy.stats import shapiro, normaltest
                if len(series) <= 5000:
                    stat, p = shapiro(series)
                    test_name = "shapiro"
                else:
                    stat, p = normaltest(series)
                    test_name = "dagostino"
                result[col] = {"is_normal": p > 0.05, "p_value": float(p), "test": test_name}
            except Exception as e:
                # fallback heuristic (skew/kurtosis) if scipy unavailable or error
                try:
                    skew = series.skew()
                    kurt = series.kurtosis()
                    is_norm = abs(skew) < 0.5 and abs(kurt) < 0.5
                    result[col] = {"is_normal": is_norm, "p_value": None, "test": "heuristic"}
                except Exception:
                    result[col] = {"is_normal": False, "p_value": None, "test": "failed"}
        return result

    def _check_correlation(self, df: pd.DataFrame) -> dict:
        """Compute correlation matrix and identify strong correlations.

        Returns a dict with keys:
          - "matrix":         nested dict of actual correlation values (tests assert this key)
          - "matrix_summary": human-readable summary string
          - "strong_pairs":   list of pairs with |corr| > 0.7
          - "n_strong":       count of strong pairs
        """
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()
        strong_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_pairs.append({
                        "col1":        col1,
                        "col2":        col2,
                        "correlation": round(float(corr_val), 3),
                        "strength":    "very_strong" if abs(corr_val) > 0.9 else "strong",
                    })

        # Build the full matrix dict — tests assert "matrix" in result
        matrix_dict = {
            col: {
                c2: round(float(corr_matrix.loc[col, c2]), 4)
                for c2 in corr_matrix.columns
            }
            for col in corr_matrix.columns
        }

        return {
            "matrix":         matrix_dict,
            "matrix_summary": f"Correlation matrix of {numeric_df.shape[1]} columns calculated.",
            "strong_pairs":   strong_pairs[:20],
            "n_strong":       len(strong_pairs),
        }

    def _check_multicollinearity(self, df: pd.DataFrame, target_col: Optional[str]) -> dict:
        """Check multicollinearity via Variance Inflation Factor (VIF).
        Drops the target column before computing VIF on numeric features.
        Returns dict with keys:
          - "multicollinearity_detected": bool
          - "high_vif_cols": {col: vif}
          - "vif_scores": {col: vif}
        Uses a 30‑second timeout and falls back if statsmodels unavailable.
        """
        try:
            import concurrent.futures
            def compute_vif():
                numeric_df = df.select_dtypes(include=np.number)
                if target_col and target_col in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[target_col])
                # Drop rows with any NA to avoid errors in VIF calculation
                clean_df = numeric_df.dropna()
                if clean_df.shape[1] == 0:
                    return {}
                X = clean_df.values
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif_dict = {}
                for i in range(X.shape[1]):
                    vif = variance_inflation_factor(X, i)
                    col_name = clean_df.columns[i]
                    vif_dict[col_name] = float(vif)
                return vif_dict
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                vif_scores = executor.submit(compute_vif).result(timeout=30)
            high_vif = {col: v for col, v in vif_scores.items() if v > 10}
            return {
                "multicollinearity_detected": bool(high_vif),
                "high_vif_cols": high_vif,
                "vif_scores": vif_scores,
            }
        except Exception as e:
            # If statsmodels not installed or timeout or any other error
            return {"skipped": "statsmodels not installed or VIF calculation failed", "multicollinearity_detected": False}


    def _check_stationarity(self, df: pd.DataFrame) -> dict:
        """Check stationarity of numeric columns using ADF test.
        Returns a dict mapping column name -> {
            "is_stationary": bool,
            "p_value": float|None,
            "test_stat": float|None,
            "used_lag": int|None,
            "critical_values": dict|None,
        }.
        Skips columns with ≤30 non‑null observations.
        If statsmodels is unavailable, returns a single entry indicating the skip.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except Exception:
            return {"skipped": "statsmodels not installed"}
        result = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) <= 30:
                continue
            try:
                adf_res = adfuller(series, autolag="AIC")
                test_stat, p_value, usedlag, nobs, crit_vals, icbest = adf_res
                is_stationary = p_value < 0.05
                result[col] = {
                    "is_stationary": is_stationary,
                    "p_value": float(p_value),
                    "test_stat": float(test_stat),
                    "used_lag": usedlag,
                    "critical_values": {k: float(v) for k, v in crit_vals.items()},
                }
            except Exception:
                # On failure for this column, mark as non‑stationary with unknown stats
                result[col] = {
                    "is_stationary": False,
                    "p_value": None,
                    "test_stat": None,
                    "used_lag": None,
                    "critical_values": None,
                }
        return result

    def _interpret_findings(self, report: dict, shape: tuple) -> str:
        import httpx

        findings_text = (
            f"Dataset shape: {shape}\n"
            f"Normality: {len([v for v in report['normality'].values() if isinstance(v, dict) and v.get('is_normal')])} "
            f"of {len(report['normality'])} columns are normal\n"
            f"Strong correlations: {report['correlation'].get('n_strong', 0)} pairs\n"
            f"Multicollinearity: {report['multicollinearity'].get('multicollinearity_detected', False)}\n"
            f"Non-stationary columns: {len([v for v in report['stationarity'].values() if isinstance(v, dict) and not v.get('is_stationary', True)])}"
        )

        model = REVIEWER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    model,
                    "messages": [
                        {"role": "system", "content": "You are a statistician. Interpret findings concisely in 3-4 sentences."},
                        {"role": "user",   "content": f"Interpret these statistical findings:\n{findings_text}"},
                    ],
                    "stream":  False,
                    "options": {"num_predict": 300, "temperature": 0.2},
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
        except Exception:
            pass
        return findings_text

    def _generate_recommendations(self, report: dict) -> list[str]:
        recs = []

        non_normal = [k for k, v in report["normality"].items() if isinstance(v, dict) and not v.get("is_normal", True)]
        if non_normal:
            recs.append(f"Apply log/sqrt transformation to non-normal columns: {', '.join(non_normal[:3])}")

        if report["multicollinearity"].get("multicollinearity_detected"):
            high_vif = list(report["multicollinearity"].get("high_vif_cols", {}).keys())
            recs.append(f"Consider removing/combining highly collinear features: {', '.join(high_vif[:3])}")

        if report["correlation"].get("n_strong", 0) > 3:
            recs.append("Use PCA or feature selection to handle multicollinearity")

        non_stationary = [k for k, v in report["stationarity"].items() if isinstance(v, dict) and not v.get("is_stationary", True)]
        if non_stationary:
            recs.append(f"Apply differencing to non-stationary columns before time-series modeling: {', '.join(non_stationary[:3])}")

        if not recs:
            recs.append("Data appears statistically well-behaved — proceed with standard modeling")

        return recs
