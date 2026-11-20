"""Gradient-boosting backend.

LightGBM is the recommended production model (it won the M5 forecasting competition and trains in
seconds on SME-sized data). Its macOS wheel needs the OpenMP runtime (``brew install libomp``), so
if LightGBM cannot be imported or loaded we fall back to scikit-learn's HistGradientBoosting
(same histogram-GBM family, Poisson loss instead of Tweedie). Force a backend with
``DATARAT_GBM=lightgbm|sklearn``.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

from .config import PlannerConfig

_LGB: Any = None
_LGB_TRIED = False


def _lightgbm() -> Any:
    global _LGB, _LGB_TRIED
    if os.environ.get("DATARAT_GBM", "").strip().lower() == "sklearn":
        return None
    if not _LGB_TRIED:
        _LGB_TRIED = True
        try:
            import lightgbm  # noqa: F401  (OSError when libomp is missing on macOS)
            _LGB = lightgbm
        except Exception:  # noqa: BLE001
            _LGB = None
    return _LGB


def backend_name() -> str:
    return "lightgbm" if _lightgbm() is not None else "sklearn-hgb"


def make_regressor(cfg: PlannerConfig) -> Any:
    """Non-negative, right-skewed daily demand -> Tweedie (LightGBM) or Poisson (sklearn) loss."""
    lgb = _lightgbm()
    if lgb is not None:
        return lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.2, n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=cfg.seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=cfg.n_estimators, learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.num_leaves, min_samples_leaf=cfg.min_child_samples, l2_regularization=1.0,
        categorical_features="from_dtype", random_state=cfg.seed)


def make_classifier(cfg: PlannerConfig) -> Any:
    lgb = _lightgbm()
    if lgb is not None:
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=max(100, cfg.n_estimators // 2), learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=cfg.seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=max(100, cfg.n_estimators // 2), learning_rate=cfg.learning_rate, max_leaf_nodes=cfg.num_leaves,
        min_samples_leaf=cfg.min_child_samples, l2_regularization=1.0, categorical_features="from_dtype",
        random_state=cfg.seed)


def fit_quiet(model: Any, X: Any, y: Any, sample_weight: Any = None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sample_weight is None:
            return model.fit(X, y)
        return model.fit(X, y, sample_weight=sample_weight)


def feature_importance(model: Any, columns: list[str], top: int = 12) -> list[dict[str, Any]]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    total = float(sum(imp)) or 1.0
    pairs = sorted(zip(columns, imp), key=lambda x: -x[1])[:top]
    return [{"feature": c, "importance": round(float(v) / total, 4)} for c, v in pairs]
