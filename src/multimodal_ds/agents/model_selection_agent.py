'''Model Selection Agent

This module implements a rule‑based agent that selects a primary model and an
ensemble of candidate models based on data characteristics derived from a
statistical report and an AutoML suggestion. The logic is deterministic and does
not involve any LLM calls.

The selected configuration is stored in the shared :class:`~multimodal_ds.core.context_pool.ContextPool`
for downstream agents to consume.
'''  # noqa: D400,D401

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pandas as pd
try:
    import optuna
    from optuna import logging as optuna_logging
    optuna_logging.set_verbosity(optuna_logging.WARNING)
except ImportError:
    optuna = None
    optuna_logging = None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_module_available(module_name: str) -> bool:
    """Return ``True`` if *module_name* can be imported.

    The function performs a lightweight import check without importing the
    module into the global namespace – useful for optional dependencies such as
    ``xgboost`` or ``lightgbm``.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# ---------------------------------------------------------------------------
# ModelSelectionAgent definition
# ---------------------------------------------------------------------------

class ModelSelectionAgent:
    """Rule‑based model selector.

    The agent inspects a :class:`pandas.DataFrame`, a statistical report and an
    AutoML suggestion to produce a dictionary describing the chosen primary model,
    an ensemble of auxiliary models, preprocessing steps, hyper‑parameter
    grid, cross‑validation strategy, scoring metric and a short rationale.

    The result is stored in the shared context pool under the key
    ``"model_selection"``.
    """

    def __init__(self, session_id: str = "default") -> None:
        """Create a new :class:`ModelSelectionAgent`.

        Parameters
        ----------
        session_id:
            Identifier for the current analysis session. The same identifier is
            used to retrieve a shared :class:`~multimodal_ds.core.context_pool.ContextPool`
            instance.
        """
        self.session_id = session_id
        # Import lazily to avoid circular imports – the pool is a singleton per
        # session.
        from multimodal_ds.core.context_pool import get_context_pool

        self.pool = get_context_pool(session_id)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def select_models(
        self,
        df: pd.DataFrame,
        target_col: str,
        stat_report: Dict[str, Any],
        automl_suggestion: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select primary and ensemble models based on data characteristics.

        The function follows a deterministic rule‑set (see the module level
        documentation for the full decision tree).
        """
        # 1. Determine task type
        task_type = automl_suggestion.get("task", "classification").lower()

        # 2. Basic shape information
        n_rows, n_cols = df.shape

        # 3. Extract statistical cues
        normality = stat_report.get("normality", {})
        non_normal_cols = [
            k for k, v in normality.items() if isinstance(v, dict) and not v.get("is_normal", True)
        ]
        multicollinearity_info = stat_report.get("multicollinearity", {})
        has_multicollinearity = multicollinearity_info.get("multicollinearity_detected", False)
        n_strong_correlations = stat_report.get("correlation", {}).get("n_strong", 0)

        # 4. Classification‑specific balance check
        is_imbalanced = False
        if task_type == "classification":
            if target_col in df.columns:
                value_counts = df[target_col].value_counts(normalize=True)
                is_imbalanced = value_counts.min() < 0.20

        # 5. Model selection rules
        primary_model: str = ""
        ensemble_models: List[str] = []
        rationale_parts: List[str] = []
        scoring_metric: str = ""
        cv_strategy: str = ""
        preprocessing_steps: List[str] = ["StandardScaler", "handle_missing_values"]

        # Helper flags for optional libraries
        have_xgboost = _is_module_available("xgboost")
        have_lightgbm = _is_module_available("lightgbm")
        have_imblearn = _is_module_available("imblearn")

        if task_type == "classification":
            # Primary model selection based on dataset size
            if n_rows < 1000:
                primary_model = "RandomForestClassifier"
                ensemble_models = [
                    "GradientBoostingClassifier",
                    "LogisticRegression",
                    "SVC",
                ]
            elif n_rows < 10000:
                primary_model = "XGBClassifier" if have_xgboost else "GradientBoostingClassifier"
                ensemble_models = [
                    "RandomForestClassifier",
                    "LogisticRegression",
                    "ExtraTreesClassifier",
                ]
            else:
                primary_model = "XGBClassifier" if have_xgboost else "GradientBoostingClassifier"
                if have_lightgbm:
                    ensemble_models = ["LGBMClassifier", "RandomForestClassifier", "LogisticRegression"]
                else:
                    ensemble_models = ["GradientBoostingClassifier", "RandomForestClassifier", "LogisticRegression"]

            # Imbalance handling
            if is_imbalanced:
                rationale_parts.append("target column is imbalanced – using class_weight='balanced'")
                # Some models support the class_weight argument directly; we note it in the rationale.

            # Scoring & CV strategy
            # Determine if binary classification – we approximate by checking unique values count.
            unique_targets = df[target_col].nunique() if target_col in df.columns else 2
            if unique_targets == 2:
                scoring_metric = "roc_auc"
            else:
                scoring_metric = "f1_weighted"
            cv_strategy = "stratified_kfold"
        elif task_type == "regression":
            if has_multicollinearity or n_strong_correlations > 3:
                primary_model = "Ridge"
                ensemble_models = ["ElasticNet", "GradientBoostingRegressor", "RandomForestRegressor"]
            else:
                primary_model = "GradientBoostingRegressor"
                ensemble_models = ["RandomForestRegressor", "Ridge", "SVR"]
            scoring_metric = "neg_root_mean_squared_error"
            cv_strategy = "kfold"
        else:
            # Fallback – treat unknown task as classification
            primary_model = "RandomForestClassifier"
            ensemble_models = ["GradientBoostingClassifier", "LogisticRegression", "SVC"]
            scoring_metric = "roc_auc"
            cv_strategy = "stratified_kfold"

        # 6. Preprocessing steps based on characteristics
        if non_normal_cols:
            preprocessing_steps.append("PowerTransformer")
        if has_multicollinearity:
            preprocessing_steps.append("PCA or drop correlated features (VIF > 10)")
        if is_imbalanced and have_imblearn:
            preprocessing_steps.append("SMOTE oversampling (imblearn)")
        elif is_imbalanced:
            preprocessing_steps.append("class_weight balanced")

        # 7. Hyperparameter grid – only include grids for models that appear in the selection.
        hyperparameter_grid: Dict[str, Any] = {}
        # Primary model grid
        if primary_model == "RandomForestClassifier":
            hyperparameter_grid[primary_model] = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }
        elif primary_model == "GradientBoostingClassifier":
            hyperparameter_grid[primary_model] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
        elif primary_model == "LogisticRegression":
            hyperparameter_grid[primary_model] = {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]}
        elif primary_model == "XGBClassifier":
            hyperparameter_grid[primary_model] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
        elif primary_model == "Ridge":
            hyperparameter_grid[primary_model] = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        elif primary_model == "GradientBoostingRegressor":
            hyperparameter_grid[primary_model] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
        # Ensembles – add grids for each if not already covered
        for model_name in ensemble_models:
            if model_name not in hyperparameter_grid:
                # Use generic grids for known classes
                if model_name in {"GradientBoostingClassifier", "GradientBoostingRegressor"}:
                    hyperparameter_grid[model_name] = {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5],
                    }
                elif "RandomForest" in model_name:
                    hyperparameter_grid[model_name] = {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                    }
                elif model_name == "LogisticRegression":
                    hyperparameter_grid[model_name] = {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]}
                elif model_name == "ElasticNet":
                    hyperparameter_grid[model_name] = {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
                # Other models (e.g., SVC, ExtraTreesClassifier) we leave empty – they will use defaults.

        # 8. Assemble result dictionary
        result: Dict[str, Any] = {
            "primary_model": primary_model,
            "ensemble_models": ensemble_models,
            "rationale": ", ".join(rationale_parts) if rationale_parts else "",
            "cv_strategy": cv_strategy,
            "scoring_metric": scoring_metric,
            "preprocessing_steps": preprocessing_steps,
+            "tuning_available": True,
+            "tuning_metadata": {},
        }

        # Store in shared context pool for downstream consumption
        self.pool.set("model_selection", result, agent="model_selection_agent")

        return result

    def generate_ensemble_code(
        self,
        selection: Dict[str, Any],
        df_varname: str = "df",
        target_col: str = "target",
    ) -> str:
        """Return a self‑contained Python script implementing the selected models.

        The generated code:
        * Imports the necessary scikit‑learn (and optional) classes.
        * Builds a ``VotingClassifier`` or ``VotingRegressor`` depending on the
          task type inferred from ``selection['primary_model']``.
        * Executes ``cross_val_score`` with the requested CV strategy and
          scoring metric.
        * Wraps the primary model in a ``GridSearchCV`` using the supplied
          ``hyperparameter_grid``.
        * Prints the best hyper‑parameters and cross‑validation statistics.
        * Saves the best estimator to ``best_model.joblib`` and, if available,
          writes feature importances to ``feature_importances.csv``.
        """
        primary = selection["primary_model"]
        ensemble = selection["ensemble_models"]
        grid = selection["hyperparameter_grid"].get(primary, {})
        cv = selection["cv_strategy"]
        scoring = selection["scoring_metric"]
        # Determine task type from primary model name prefix
        task_is_classification = "Classifier" in primary
        voting_class = "VotingClassifier" if task_is_classification else "VotingRegressor"

        # Build import lines – include only needed classes
        imports: List[str] = []
        # Core sklearn utilities
        imports.append("from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold")
        imports.append("from sklearn.metrics import classification_report, r2_score, mean_squared_error")
        imports.append("import joblib")
        import pandas as pd
        imports.append("import numpy as np")
        # Primary model import
        primary_mod = primary.replace("Classifier", "").replace("Regressor", "")
        imports.append(f"from sklearn.ensemble import {primary}")
        # Ensemble model imports – deduplicate
        for mdl in ensemble:
            # Guard against duplicates with primary
            if mdl == primary:
                continue
            # Simple heuristic: most sklearn models live in ensemble, linear_model, svm, etc.
            if "Forest" in mdl or "GradientBoosting" in mdl or "AdaBoost" in mdl or "Bagging" in mdl:
                imports.append(f"from sklearn.ensemble import {mdl}")
            elif "Regression" in mdl:
                imports.append(f"from sklearn.ensemble import {mdl}")
            elif "SVC" in mdl or "SVR" in mdl:
                imports.append(f"from sklearn.svm import {mdl}")
            elif "LogisticRegression" in mdl or "LinearRegression" in mdl or "Ridge" in mdl or "ElasticNet" in mdl:
                imports.append(f"from sklearn.linear_model import {mdl}")
            else:
                # Fallback generic import
                imports.append(f"# TODO: import {mdl} from the correct submodule")
        # Remove potential duplicate import lines
        imports = list(dict.fromkeys(imports))

        # Build the voting estimator dictionary string
        voting_estimators = []
        voting_estimators.append(f"('{primary.lower()}', {primary}())")
        for idx, mdl in enumerate(ensemble, start=1):
            var_name = mdl.lower()
            voting_estimators.append(f"('{var_name}{idx}', {mdl}())")
        voting_dict_str = ", ".join(voting_estimators)

        # CV object based on strategy
        if cv == "stratified_kfold":
            cv_obj = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
        elif cv == "kfold":
            cv_obj = "KFold(n_splits=5, shuffle=True, random_state=42)"
        elif cv == "time_series_split":
            cv_obj = "TimeSeriesSplit(n_splits=5)"
        else:
            cv_obj = "KFold(n_splits=5, shuffle=True, random_state=42)"

        # Assemble the script
        lines: List[str] = []
        lines.extend(imports)
        lines.append("")
        lines.append(f"# Assuming the data is stored in a pandas DataFrame named '{df_varname}'")
        lines.append(f"X = {df_varname}.drop(columns=['{target_col}'])")
        lines.append(f"y = {df_varname}['{target_col}']")
        lines.append("")
        # Primary model with GridSearchCV if grid provided
        if grid:
            lines.append(f"primary_estimator = GridSearchCV({primary}(), param_grid={grid}, cv=3, scoring='{scoring}')")
        else:
            lines.append(f"primary_estimator = {primary}()")
        lines.append("")
        # Voting ensemble
        lines.append(f"voting_estimator = {voting_class}(estimators=[{voting_dict_str}], voting='soft' if '{voting_class}' == 'VotingClassifier' else 'hard')")
        lines.append("")
        # Cross‑validation evaluation
        lines.append(f"cv = {cv_obj}")
        lines.append(f"scores = cross_val_score(voting_estimator, X, y, cv=cv, scoring='{scoring}')")
        lines.append("print(f'Cross‑val {scoring}: {np.mean(scores):.4f} ± {np.std(scores):.4f}')")
        lines.append("")
        # Fit primary model (with hyper‑parameter search) and then ensemble
        lines.append("primary_estimator.fit(X, y)")
        lines.append("best_model = primary_estimator.best_estimator_ if hasattr(primary_estimator, 'best_estimator_') else primary_estimator")
        lines.append("voting_estimator.fit(X, y)")
        lines.append("# Save the best primary model")
        lines.append("joblib.dump(best_model, 'best_model.joblib')")
        lines.append("# Save voting ensemble as well")
        lines.append("joblib.dump(voting_estimator, 'voting_ensemble.joblib')")
        lines.append("")
        # Reporting
        if task_is_classification:
            lines.append("y_pred = voting_estimator.predict(X)")
            lines.append("print(classification_report(y, y_pred))")
        else:
            lines.append("y_pred = voting_estimator.predict(X)")
            lines.append("rmse = mean_squared_error(y, y_pred, squared=False)")
            lines.append("print(f'RMSE: {rmse:.4f}')")
            lines.append("r2 = r2_score(y, y_pred)")
            lines.append("print(f'R^2: {r2:.4f}')")
        lines.append("")
        # Feature importances if present
        lines.append("if hasattr(best_model, 'feature_importances_'):")
        lines.append(f"pd.DataFrame({{'feature': X.columns,'importance': best_model.feature_importances_}}).sort_values('importance', ascending=False).to_csv('feature_importances.csv', index=False)")
        lines.append("else:")
        lines.append("    print('Model does not expose feature_importances_')")
        lines.append("")
        script = "\n".join(lines)
        return script

    def tune_model(self, model_name: str, X, y, cv_strategy: str, scoring: str, search_space: Dict[str, Any], n_trials: int = 30) -> Dict[str, Any]:
        """Tune a single model using Optuna.

        Returns a dictionary with best_params, best_value and the Optuna study.
        """
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        import importlib

        def get_model_class(name: str):
            # Map common model names to sklearn classes
            if name in {"RandomForestClassifier", "GradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier", "VotingClassifier"}:
                module = importlib.import_module('sklearn.ensemble')
            elif name in {"RandomForestRegressor", "GradientBoostingRegressor", "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "VotingRegressor"}:
                module = importlib.import_module('sklearn.ensemble')
            elif name in {"LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet"}:
                module = importlib.import_module('sklearn.linear_model')
            elif name in {"SVC", "SVR"}:
                module = importlib.import_module('sklearn.svm')
            else:
                raise ValueError(f"Unsupported model {name}")
            return getattr(module, name)

        def objective(trial):
            params = {}
            for param, spec in search_space.items():
                p_type = spec.get('type')
                if p_type == 'int':
                    params[param] = trial.suggest_int(param, spec['low'], spec['high'])
                elif p_type == 'float':
                    params[param] = trial.suggest_float(param, spec['low'], spec['high'], log=spec.get('log', False))
                elif p_type == 'categorical':
                    params[param] = trial.suggest_categorical(param, spec['choices'])
            ModelClass = get_model_class(model_name)
            model = ModelClass(**params)
            # Determine CV object
            if cv_strategy == "stratified_kfold":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            # For metrics where higher is better (e.g., roc_auc, f1), maximize; otherwise minimize
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return {"best_params": study.best_params, "best_value": study.best_value, "study": study}

    def tune_all_models(self, X, y, selection: Dict[str, Any], n_trials: int = 30) -> Dict[str, Any]:
        """Tune primary and ensemble models and store results in the context pool.

        Returns a dict mapping model name to its tuning result.
        """
        tuning_results = {}
        # Define simple default search spaces mirroring previous static grids
        default_spaces = {
            "RandomForestClassifier": {"n_estimators": {"type": "int", "low": 50, "high": 300}, "max_depth": {"type": "int", "low": 5, "high": 30}},
            "GradientBoostingClassifier": {"n_estimators": {"type": "int", "low": 50, "high": 300}, "learning_rate": {"type": "float", "low": 0.01, "high": 0.2}, "max_depth": {"type": "int", "low": 3, "high": 10}},
            "LogisticRegression": {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}, "penalty": {"type": "categorical", "choices": ["l2"]}},
            "XGBClassifier": {"n_estimators": {"type": "int", "low": 50, "high": 300}, "learning_rate": {"type": "float", "low": 0.01, "high": 0.2}, "max_depth": {"type": "int", "low": 3, "high": 10}},
            "Ridge": {"alpha": {"type": "float", "low": 0.1, "high": 100.0}},
            "GradientBoostingRegressor": {"n_estimators": {"type": "int", "low": 50, "high": 300}, "learning_rate": {"type": "float", "low": 0.01, "high": 0.2}, "max_depth": {"type": "int", "low": 3, "high": 10}},
        }
        models_to_tune = [selection["primary_model"]] + selection.get("ensemble_models", [])
        for mdl in models_to_tune:
            space = default_spaces.get(mdl, {})
            if not space:
                continue
            result = self.tune_model(mdl, X, y, selection.get("cv_strategy", "kfold"), selection.get("scoring_metric", "r2"), space, n_trials=n_trials)
            tuning_results[mdl] = result
        # Determine best overall model
        best_overall_model = None
        best_overall_score = float('-inf')
        for mdl, res in tuning_results.items():
            val = res.get('best_value')
            if val is not None and val > best_overall_score:
                best_overall_score = val
                best_overall_model = mdl
        # Store summary in results
        tuning_results['best_overall_model'] = best_overall_model
        tuning_results['best_overall_score'] = best_overall_score

    def generate_tuned_ensemble_code(
            self,
            selection: Dict[str, Any],
            tuning_results: Dict[str, Any],
            df_varname: str = "df",
            target_col: str = "target",
        ) -> str:
        """Generate Python script using tuned hyper‑parameters for the primary model.
        """
        primary = selection["primary_model"]
        ensemble = selection.get("ensemble_models", [])
        best_params = tuning_results.get(primary, {}).get("best_params", {})
        cv = selection.get("cv_strategy", "kfold")
        scoring = selection.get("scoring_metric", "r2")
        task_is_classification = "Classifier" in primary
        voting_class = "VotingClassifier" if task_is_classification else "VotingRegressor"
        # Build imports similar to generate_ensemble_code
        imports: List[str] = []
        imports.append("from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score")
        imports.append("from sklearn.metrics import classification_report, r2_score, mean_squared_error")
        imports.append("import joblib, pandas as pd, numpy as np")
        # Primary model import with params
        primary_mod = primary.replace("Classifier", "").replace("Regressor", "")
        imports.append(f"from sklearn.ensemble import {primary}")
        for mdl in ensemble:
            if mdl == primary:
                continue
            if "Forest" in mdl or "GradientBoosting" in mdl or "AdaBoost" in mdl or "Bagging" in mdl:
                imports.append(f"from sklearn.ensemble import {mdl}")
            elif "Regression" in mdl:
                imports.append(f"from sklearn.ensemble import {mdl}")
            elif mdl in {"SVC", "SVR"}:
                imports.append(f"from sklearn.svm import {mdl}")
            elif mdl in {"LogisticRegression", "LinearRegression", "Ridge", "ElasticNet"}:
                imports.append(f"from sklearn.linear_model import {mdl}")
        imports = list(dict.fromkeys(imports))
        lines: List[str] = []
        lines.extend(imports)
        lines.append("")
        lines.append(f"# Data preparation assuming DataFrame '{df_varname}'")
        lines.append(f"X = {df_varname}.drop(columns=['{target_col}'])")
        lines.append(f"y = {df_varname}['{target_col}']")
        lines.append("")
        # Primary model instantiated with tuned params
        if best_params:
            lines.append(f"primary_estimator = {primary}(**{best_params})")
        else:
            lines.append(f"primary_estimator = {primary}()")
        lines.append("")
        # Voting ensemble
        voting_estimators = [f"('{primary.lower()}', primary_estimator)"]
        for idx, mdl in enumerate(ensemble, start=1):
            var_name = mdl.lower()
            voting_estimators.append(f"('{var_name}{idx}', {mdl}())")
        voting_dict_str = ", ".join(voting_estimators)
        lines.append(f"voting_estimator = {voting_class}(estimators=[{voting_dict_str}], voting='soft' if '{voting_class}' == 'VotingClassifier' else 'hard')")
        lines.append("")
        # CV evaluation
        if cv == "stratified_kfold":
            cv_obj = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
        else:
            cv_obj = "KFold(n_splits=5, shuffle=True, random_state=42)"
        lines.append(f"cv = {cv_obj}")
        lines.append(f"scores = cross_val_score(voting_estimator, X, y, cv=cv, scoring='{scoring}')")
        lines.append("print(f'Cross‑val {scoring}: {np.mean(scores):.4f} ± {np.std(scores):.4f}')")
        lines.append("")
        # Fit models
        lines.append("primary_estimator.fit(X, y)")
        lines.append("voting_estimator.fit(X, y)")
        lines.append("joblib.dump(primary_estimator, 'primary_model.joblib')")
        lines.append("joblib.dump(voting_estimator, 'voting_ensemble.joblib')")
        lines.append("")
        if task_is_classification:
            lines.append("y_pred = voting_estimator.predict(X)")
            lines.append("print(classification_report(y, y_pred))")
        else:
            lines.append("y_pred = voting_estimator.predict(X)")
            lines.append("rmse = mean_squared_error(y, y_pred, squared=False)")
            lines.append("print(f'RMSE: {rmse:.4f}')")
            lines.append("r2 = r2_score(y, y_pred)")
            lines.append("print(f'R^2: {r2:.4f}')")
        lines.append("")
        lines.append("if hasattr(primary_estimator, 'feature_importances_'):")
        lines.append("    pd.DataFrame({'feature': X.columns, 'importance': primary_estimator.feature_importances_}).sort_values('importance', ascending=False).to_csv('feature_importances.csv', index=False)")
        lines.append("else:")
        lines.append("    print('Model does not expose feature_importances_')")
        lines.append("")
        script = "\n".join(lines)
        return script
