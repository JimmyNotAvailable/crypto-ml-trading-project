import os
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Default path: same folder as this file
_DEFAULT_PATH = Path(__file__).parent / "crypto_models_production.pkl"

# Allow override by environment variable
_ENV_PATH = os.getenv("MODELS_PROD_PATH")

_CACHE: Dict[str, Any] | None = None
_CACHE_MTIME: float | None = None


def _resolve_prod_path() -> Path:
    p = Path(_ENV_PATH) if _ENV_PATH else _DEFAULT_PATH
    return p


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_joblib(path: Path):
    try:
        import joblib  # lazy import
        return joblib.load(path)
    except Exception:
        raise


def load_production_models():
    """Load production package with simple cache and hot-reload on file change."""
    global _CACHE, _CACHE_MTIME
    path = _resolve_prod_path()
    if not path.exists():
        raise FileNotFoundError(f"Production model package not found: {path}")
    mtime = path.stat().st_mtime
    if _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE
    # Try pickle first, then joblib
    try:
        pkg = _load_pickle(path)
    except Exception:
        pkg = _load_joblib(path)
    _CACHE, _CACHE_MTIME = pkg, mtime
    return pkg


def predict_price(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict price and trend using the production package.

    Input: features_dict (dict of feature_name -> value)
    Output: {
      'predicted_price': float,
      'trend': int,
      'confidence': 'high'|'medium'|'low',
      'model_name': str,
      'metrics': {'r2': float|None, 'mae': float|None},
    }
    """
    pkg = load_production_models()

    # Defensive extraction
    models = (pkg.get("models") or {}) if isinstance(pkg, dict) else {}
    price_model = models.get("price_predictor")
    trend_model = models.get("trend_classifier")
    if price_model is None or trend_model is None:
        raise KeyError("Production package missing required models: price_predictor/trend_classifier")

    feats = (pkg.get("feature_cols") or []) if isinstance(pkg, dict) else []
    df = pd.DataFrame([features_dict])
    if feats:
        df = df.reindex(columns=feats, fill_value=0)
    # Coerce to numeric to be safe
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Price prediction
    price_raw = price_model.predict(df)
    try:
        price = float(price_raw[0])
    except Exception:
        price = float(price_raw)

    # Trend prediction (robust to different return types)
    t_pred = getattr(trend_model, "predict", lambda X: [1])(df)
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(t_pred, _pd.DataFrame):
            t_pred = t_pred.iloc[:, 0].to_numpy()
        elif isinstance(t_pred, _pd.Series):
            t_pred = t_pred.to_numpy()
        if isinstance(t_pred, _np.ndarray):
            trend = int(t_pred.flat[0])
        elif isinstance(t_pred, (list, tuple)):
            trend = int(t_pred[0])
        else:
            trend = int(t_pred)
    except Exception:
        trend = 1

    perf = (pkg.get("performance") or {}) if isinstance(pkg, dict) else {}
    r2 = perf.get("price_predictor_r2")
    mae = perf.get("price_predictor_mae")
    conf = "high" if (isinstance(r2, (int, float)) and r2 > 0.7) else ("medium" if r2 else "medium")

    meta = (pkg.get("metadata") or {}) if isinstance(pkg, dict) else {}
    best_price = meta.get("best_price_model", "?")
    best_trend = meta.get("best_trend_model", "?")

    return {
        "predicted_price": price,
        "trend": trend,
        "confidence": conf,
        "model_name": f"{best_price}+{best_trend}",
        "metrics": {"r2": r2, "mae": mae},
    }
