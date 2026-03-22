import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class FusionConfig:
    # none/mom/yoy
    transform_type: str
    # zscore/minmax
    standardize: str
    # equal/pca/entropy
    fusion_method: str
    inverse_columns: List[str]
    smoothing_window: int


def _parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        # Try common alternatives
        for cand in ["Date", "DATE", "month", "Month", "time", "Time", "ds"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    if "date" not in df.columns:
        raise ValueError("CSV must contain a date column named `date` (or Date/DATE/month/etc.).")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Unify to month-start so forecasting (freq="MS") and time interpolation are consistent.
    # pandas 对 PeriodArray 的 to_timestamp 不同版本对 freq 参数支持不一致；
    # 这里统一取“月初”即可。
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    df = df.sort_values("date")
    return df


def load_user_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    df = pd.read_csv(io.BytesIO(raw))
    return _parse_date_col(df)


def coerce_numeric_and_clean(
    df: pd.DataFrame,
    indicator_cols: List[str],
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    df = df.copy()
    lo_q, hi_q = clip_quantiles

    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Interpolate missing values in time order
        df[col] = df[col].interpolate(method="time", limit_direction="both")

        lo, hi = df[col].quantile(lo_q), df[col].quantile(hi_q)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def transform_indicators(df: pd.DataFrame, indicator_cols: List[str], transform_type: str) -> pd.DataFrame:
    df = df.copy()
    if transform_type == "none":
        return df
    if transform_type not in {"mom", "yoy"}:
        raise ValueError("transform_type must be one of: none/mom/yoy")

    # Assume monthly frequency for yoy.
    periods = 1 if transform_type == "mom" else 12
    for col in indicator_cols:
        df[col] = df[col].pct_change(periods=periods)
    return df


def standardize_matrix(X: pd.DataFrame, method: str) -> pd.DataFrame:
    X = X.copy()
    if method == "zscore":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0).replace(0.0, np.nan)
        X = (X - mu) / sigma
        return X.fillna(0.0)

    if method == "minmax":
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        denom = (hi - lo).replace(0.0, np.nan)
        X = (X - lo) / denom
        return X.fillna(0.0)

    raise ValueError("standardize must be zscore or minmax")


def compute_entropy_weights(X_minmax: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    X_minmax: (n_samples, n_features), values in [0,1] ideally.
    Entropy weights require non-negative numbers; we re-normalize per feature.
    """
    X = np.asarray(X_minmax, dtype=float)
    # Make strictly positive
    X = np.clip(X, eps, 1.0)

    # Probability distribution per feature across time
    col_sums = X.sum(axis=0, keepdims=True)
    P = X / col_sums

    # Entropy
    k = 1.0 / np.log(X.shape[0]) if X.shape[0] > 1 else 0.0
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1.0 - E
    d = np.clip(d, eps, None)
    w = d / d.sum()
    return w


def fuse_indicators(X: pd.DataFrame, config: FusionConfig) -> Tuple[pd.Series, Dict[str, float]]:
    inv = set([c for c in config.inverse_columns])
    X_use = X.copy()
    for c in X_use.columns:
        if c in inv:
            # When higher means worse, flip the sign so "higher composite" still indicates "better".
            X_use[c] = -X_use[c]

    # Normalize/standardize before fusion.
    X_std = standardize_matrix(X_use, method=config.standardize)

    if config.fusion_method == "equal":
        weights = {c: 1.0 / X_std.shape[1] for c in X_std.columns}
        comp = X_std.mean(axis=1)
        return comp, weights

    if config.fusion_method == "pca":
        # Use first principal component loadings magnitude as weights.
        pca = PCA(n_components=1, random_state=0)
        pca.fit(X_std.values)
        loadings = pca.components_[0]
        abs_load = np.abs(loadings)
        if abs_load.sum() == 0:
            w = np.ones_like(abs_load) / len(abs_load)
        else:
            w = abs_load / abs_load.sum()
        weights = {c: float(w[i]) for i, c in enumerate(X_std.columns)}
        comp = np.dot(X_std.values, w)
        return pd.Series(comp, index=X_std.index), weights

    if config.fusion_method == "entropy":
        # For entropy we need non-negative; use minmax to [0,1] (recompute if needed).
        X_pos = standardize_matrix(X_use, method="minmax")
        w_vec = compute_entropy_weights(X_pos.values)
        weights = {c: float(w_vec[i]) for i, c in enumerate(X_pos.columns)}
        comp = np.dot(X_pos.values, w_vec)
        return pd.Series(comp, index=X_pos.index), weights

    raise ValueError("fusion_method must be equal/pca/entropy")


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    window = int(window)
    if window <= 1:
        return s
    return s.rolling(window=window, center=False).mean()


def classify_cycle_states(
    composite: pd.Series,
    ma_window: int = 6,
    band_multiplier: float = 0.3,
) -> pd.DataFrame:
    """
    Produce simple 4-phase classification based on:
    deviation from moving average and slope sign.

    Expansion: deviation>0 & slope>0
    Peak: deviation>0 & slope<=0
    Contraction: deviation<=0 & slope<0
    Trough: deviation<=0 & slope>=0
    """
    comp = composite.copy().astype(float)
    ma = comp.rolling(window=ma_window, min_periods=max(3, ma_window // 2)).mean()
    dev = comp - ma
    slope = comp.diff()

    band = band_multiplier * dev.std(ddof=0) if dev.notna().any() else 0.0
    # Use band to avoid flickering around MA.
    dev_pos = dev > band
    dev_neg = dev < -band

    state: List[str] = []
    for i in range(len(comp)):
        if dev_pos.iloc[i]:
            state.append("Expansion" if slope.iloc[i] > 0 else "Peak")
        elif dev_neg.iloc[i]:
            state.append("Contraction" if slope.iloc[i] < 0 else "Trough")
        else:
            state.append("Expansion" if slope.iloc[i] >= 0 else "Contraction")

    out = pd.DataFrame(
        {
            "date": comp.index,
            "composite": comp.values,
            "ma": ma.values,
            "deviation": dev.values,
            "slope": slope.values,
            "state": state,
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def forecast_composite(composite: pd.Series, horizon_months: int) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using Holt-Winters.

    Returns:
      forecast series (future index), and None for intervals (kept optional).
    """
    y = composite.dropna()
    if len(y) < 18:
        # Too short for seasonal model; fallback to linear trend in index space.
        t = np.arange(len(y))
        coef = np.polyfit(t, y.values, deg=1)
        fut_t = np.arange(len(y), len(y) + horizon_months)
        yhat = coef[0] * fut_t + coef[1]
        last_date = y.index.max()
        idx = pd.date_range(last_date, periods=horizon_months + 1, freq="MS")[1:]
        return pd.Series(yhat, index=idx, name="forecast"), None

    seasonal_periods = 12  # monthly data
    seasonal = "add"
    trend = "add"
    try:
        model = ExponentialSmoothing(
            y.values,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=True)
        yhat = fit.forecast(horizon_months)
        last_date = y.index.max()
        idx = pd.date_range(last_date, periods=horizon_months + 1, freq="MS")[1:]
        return pd.Series(yhat, index=idx, name="forecast"), None
    except Exception:
        # Fallback to non-seasonal exponential smoothing
        model = ExponentialSmoothing(y.values, trend=trend, seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        yhat = fit.forecast(horizon_months)
        last_date = y.index.max()
        idx = pd.date_range(last_date, periods=horizon_months + 1, freq="MS")[1:]
        return pd.Series(yhat, index=idx, name="forecast"), None


STATE_CN_MAP = {"Expansion": "扩张", "Peak": "滞涨/高点", "Contraction": "收缩", "Trough": "低谷"}


def create_sample_data(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="MS")

    # Synthetic “macro-like” indicators with common cycle component
    t = np.arange(n)
    cycle = np.sin(2 * np.pi * t / 48.0) + 0.4 * np.sin(2 * np.pi * t / 24.0)
    noise = lambda scale: rng.normal(0, scale, size=n)

    df = pd.DataFrame(
        {
            "date": idx,
            "gdp": 100 + 5 * cycle + noise(1.5),
            "industrial_production": 50 + 3.5 * cycle + noise(1.2),
            "cpi": 2 + 0.5 * cycle + noise(0.15),
            "m2": 200 + 2.8 * cycle + noise(2.0),
            "credit": 80 + 3.0 * cycle + noise(1.8),
            "unemployment": 5.5 - 0.3 * cycle + noise(0.12),  # inverse example
            "fx": 6.8 + 0.15 * noise(1.0) + 0.05 * (-cycle),  # weaker RMB ~ contraction
        }
    )

    # Add missing and outliers
    miss_idx = rng.choice(n, size=max(3, n // 15), replace=False)
    out_idx = rng.choice(n, size=max(2, n // 30), replace=False)
    for col in ["gdp", "industrial_production", "cpi", "m2", "credit", "unemployment", "fx"]:
        df.loc[miss_idx, col] = np.nan
    df.loc[out_idx, "credit"] *= 1.3
    return df


def run_analysis(
    df: pd.DataFrame,
    *,
    config: FusionConfig,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
    horizon_months: int = 3,
    ma_window: int = 6,
    band_multiplier: float = 0.3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, object]:
    df = df.copy()
    df = _parse_date_col(df)
    df = df.sort_values("date")

    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["date"] >= start_dt]
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df[df["date"] <= end_dt]

    if df.empty:
        raise ValueError("No data after applying date range filter.")

    indicator_cols = [c for c in df.columns if c != "date"]
    if len(indicator_cols) < 2:
        raise ValueError("CSV must include at least 2 indicator columns besides `date`.")

    # Use date as index for consistent time interpolation and forecasting.
    df = df.set_index("date", drop=False)
    # If CSV contains multiple records in the same month, aggregate by mean.
    df = df.groupby(level=0).mean(numeric_only=True)

    # 1) Cleaning (missing/outliers)
    df_clean = coerce_numeric_and_clean(df, indicator_cols, clip_quantiles=clip_quantiles)

    # 2) Transform (optional)
    df_tx = transform_indicators(df_clean, indicator_cols, config.transform_type)
    df_tx[indicator_cols] = df_tx[indicator_cols].interpolate(method="time", limit_direction="both")
    df_tx[indicator_cols] = df_tx[indicator_cols].fillna(0.0)

    # 3) Fusion
    composite_raw, weights = fuse_indicators(df_tx[indicator_cols], config=config)
    composite = smooth_series(composite_raw, config.smoothing_window)

    # 4) States on historical composite
    states_df = classify_cycle_states(composite, ma_window=ma_window, band_multiplier=band_multiplier).sort_values(
        "date"
    )
    states_df["state_cn"] = states_df["state"].map(STATE_CN_MAP)

    # 5) Forecast future composite
    forecast_series, _intervals = forecast_composite(composite, horizon_months=horizon_months)

    # Classify future states by re-running the same heuristic on concatenated series.
    comp_all = pd.concat([composite, forecast_series])
    states_all = classify_cycle_states(comp_all, ma_window=ma_window, band_multiplier=band_multiplier)
    future_states = states_all[states_all["date"].isin(forecast_series.index)].copy().sort_values("date")
    future_states["state_cn"] = future_states["state"].map(STATE_CN_MAP)
    future_states = future_states.rename(columns={"composite": "forecast_composite"})

    # JSON-friendly serialization
    def _series_to_list(series: pd.Series, value_key: str) -> List[Dict[str, object]]:
        s = series.copy()
        s = s.sort_index()
        return [{"date": str(idx.date()), value_key: float(val)} for idx, val in zip(s.index, s.values)]

    out = {
        "indicator_columns": indicator_cols,
        "fusion_config": config.__dict__,
        "clip_quantiles": list(clip_quantiles),
        "weights": weights,
        "composite_history": _series_to_list(composite, "composite"),
        "states_history": [
            {
                "date": str(row["date"].date()),
                "composite": float(row["composite"]),
                "state": row["state"],
                "state_cn": row["state_cn"],
            }
            for _, row in states_df.iterrows()
        ],
        "forecast": _series_to_list(forecast_series, "forecast_composite"),
        "future_states": [
            {
                "date": str(row["date"].date()),
                "forecast_composite": float(row["forecast_composite"]),
                "state": row["state"],
                "state_cn": row["state_cn"],
            }
            for _, row in future_states.iterrows()
        ],
    }
    return out

