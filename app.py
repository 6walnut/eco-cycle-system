import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


st.set_page_config(page_title="China Macro Economic Cycle - State Analysis", layout="wide")


@dataclass
class FusionConfig:
    transform_type: str  # none/mom/yoy
    standardize: str  # zscore/minmax
    fusion_method: str  # equal/pca/entropy
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
    # Guess delimiter by pandas
    df = pd.read_csv(io.BytesIO(raw))
    df = _parse_date_col(df)
    return df


def coerce_numeric_and_clean(
    df: pd.DataFrame,
    indicator_cols: List[str],
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    df = df.copy()
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Interpolate missing values in time order
        df[col] = df[col].interpolate(method="time", limit_direction="both")
        lo, hi = df[col].quantile(clip_quantiles[0]), df[col].quantile(clip_quantiles[1])
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
        X = X.fillna(0.0)
        return X
    if method == "minmax":
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        denom = (hi - lo).replace(0.0, np.nan)
        X = (X - lo) / denom
        X = X.fillna(0.0)
        return X
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

    state = []
    for i in range(len(comp)):
        if dev_pos.iloc[i]:
            # Above MA
            state.append("Expansion" if slope.iloc[i] > 0 else "Peak")
        elif dev_neg.iloc[i]:
            # Below MA
            state.append("Contraction" if slope.iloc[i] < 0 else "Trough")
        else:
            # Near MA: decide using slope only
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


def forecast_composite(
    composite: pd.Series,
    horizon_months: int,
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using Holt-Winters.
    Returns: forecast series (future index) and None for intervals (kept optional).
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

    # Monthly seasonal (period=12).
    seasonal_periods = 12
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


def make_phase_plot(states_df: pd.DataFrame) -> go.Figure:
    color_map = {
        "Expansion": "#1f77b4",
        "Peak": "#ff7f0e",
        "Contraction": "#d62728",
        "Trough": "#2ca02c",
    }
    fig = go.Figure()
    for s in ["Expansion", "Peak", "Contraction", "Trough"]:
        d = states_df[states_df["state"] == s]
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["composite"],
                mode="lines+markers",
                name=s,
                marker=dict(size=5),
                line=dict(width=2, color=color_map[s]),
            )
        )
    fig.update_layout(
        title="Composite Economic Cycle Index and Phase States",
        xaxis_title="Date",
        yaxis_title="Composite",
        legend_title="Phase",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def make_indicator_contrib_plot(
    last_std: pd.Series,
    weights: Dict[str, float],
) -> go.Figure:
    # contribution proxy: standardized * weight
    contrib = {}
    for col, w in weights.items():
        contrib[col] = float(last_std[col]) * float(w)
    df = pd.DataFrame({"indicator": list(contrib.keys()), "contribution": list(contrib.values())})
    df = df.sort_values("contribution", ascending=False)
    fig = px.bar(df, x="indicator", y="contribution", title="Last Period Indicator Contributions (Proxy)")
    fig.update_layout(template="plotly_white", xaxis_title="Indicator", yaxis_title="Contribution proxy")
    return fig


def create_sample_data(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="MS")
    # Synthetic “macro-like” indicators with common cycle component
    t = np.arange(n)
    cycle = np.sin(2 * np.pi * t / 48.0) + 0.4 * np.sin(2 * np.pi * t / 24.0)
    noise = lambda scale: rng.normal(0, scale, size=n)

    gdp_proxy = 100 + 5 * cycle + noise(1.5)
    ind_prod = 50 + 3.5 * cycle + noise(1.2)
    cpi = 2 + 0.5 * cycle + noise(0.15)
    m2 = 200 + 2.8 * cycle + noise(2.0)
    credit = 80 + 3.0 * cycle + noise(1.8)
    unemployment = 5.5 - 0.3 * cycle + noise(0.12)  # inverse example
    fx = 6.8 + 0.15 * noise(1.0) + 0.05 * (-cycle)  # weaker RMB ~ contraction

    df = pd.DataFrame(
        {
            "date": idx,
            "gdp": gdp_proxy,
            "industrial_production": ind_prod,
            "cpi": cpi,
            "m2": m2,
            "credit": credit,
            "unemployment": unemployment,
            "fx": fx,
        }
    )

    # Add missing and outliers
    miss_idx = rng.choice(n, size=max(3, n // 15), replace=False)
    out_idx = rng.choice(n, size=max(2, n // 30), replace=False)
    for col in ["gdp", "industrial_production", "cpi", "m2", "credit", "unemployment", "fx"]:
        df.loc[miss_idx, col] = np.nan
    df.loc[out_idx, "credit"] *= 1.3

    return df


def main():
    st.title("China Macro Economic Cycle State Analysis & Visualization System")
    st.caption("Data cleaning -> Multi-indicator fusion -> Cycle phase classification -> 3-6 months forecast")

    # Sidebar controls
    st.sidebar.header("1) Data and Pre-processing")
    st.sidebar.markdown("Upload your CSV or use generated sample data.")

    uploaded = st.sidebar.file_uploader("Upload CSV (columns: `date` + indicators)", type=["csv"])
    use_sample = uploaded is None

    if use_sample:
        df = create_sample_data()
        st.sidebar.success("Using sample data (for demo).")
    else:
        df = load_user_csv(uploaded)
        st.sidebar.success(f"Loaded rows: {len(df)}")

    df = df.copy()
    df = df.sort_values("date")

    indicator_cols = [c for c in df.columns if c != "date"]
    st.sidebar.write("Indicators detected:", indicator_cols)

    st.sidebar.subheader("Transform")
    transform_type = st.sidebar.selectbox("Transform macro indicators", ["none", "mom", "yoy"], index=1)

    st.sidebar.subheader("Cleaning")
    clip_lo = st.sidebar.slider("Clip lower quantile", 0.0, 0.05, 0.01, 0.005)
    clip_hi = st.sidebar.slider("Clip upper quantile", 0.95, 1.0, 0.99, 0.005)

    st.sidebar.subheader("Standardization")
    standardize = st.sidebar.selectbox("Standardize method", ["zscore", "minmax"], index=0)

    st.sidebar.subheader("Fusion")
    fusion_method = st.sidebar.selectbox("Fusion method", ["equal", "pca", "entropy"], index=2)
    smoothing_window = st.sidebar.slider("Smoothing window (months)", 1, 12, 3, 1)

    inverse_columns = st.sidebar.multiselect(
        "Inverse indicators (when higher means worse; will be sign-flipped)",
        options=indicator_cols,
        default=["unemployment"],
    )

    horizon_months = st.sidebar.slider("Forecast horizon (months)", 3, 6, 3, 1)
    ma_window = st.sidebar.slider("MA window for phase classification", 3, 12, 6, 1)
    band_multiplier = st.sidebar.slider("Devs band multiplier (anti-noise)", 0.0, 1.0, 0.3, 0.05)

    # Date range
    st.sidebar.subheader("Time range")
    min_date, max_date = df["date"].min(), df["date"].max()
    start_date, end_date = st.sidebar.date_input(
        "Start/End",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    # Use date as index for consistent time interpolation and forecasting.
    df = df.set_index("date", drop=False)
    # If CSV contains multiple records in the same month, aggregate by mean.
    df = df.groupby(level=0).mean(numeric_only=True)

    # Build standard dataset: clean + transform + fuse
    config = FusionConfig(
        transform_type=transform_type,
        standardize=standardize,
        fusion_method=fusion_method,
        inverse_columns=inverse_columns,
        smoothing_window=smoothing_window,
    )

    # 1. Missing/outliers cleaning
    df_clean = coerce_numeric_and_clean(df, indicator_cols, clip_quantiles=(clip_lo, clip_hi))

    # 2. Transform to growth/changes (optional)
    df_tx = transform_indicators(df_clean, indicator_cols, config.transform_type)

    # Handle transform-induced NaNs
    df_tx[indicator_cols] = df_tx[indicator_cols].interpolate(method="time", limit_direction="both")
    df_tx[indicator_cols] = df_tx[indicator_cols].fillna(0.0)

    # 3. Multi-indicator fusion
    composite_raw, weights = fuse_indicators(df_tx[indicator_cols], config=config)
    composite = smooth_series(composite_raw, config.smoothing_window)

    states_df = classify_cycle_states(composite, ma_window=ma_window, band_multiplier=band_multiplier)
    states_df = states_df.sort_values("date")
    states_df["state_cn"] = states_df["state"].map(
        {"Expansion": "扩张", "Peak": "滞涨/高点", "Contraction": "收缩", "Trough": "低谷"}
    )

    # 4. Forecast
    forecast_series, intervals = forecast_composite(composite, horizon_months=horizon_months)

    # Classify future using same logic on concatenated series
    comp_all = pd.concat([composite, forecast_series])
    states_all = classify_cycle_states(comp_all, ma_window=ma_window, band_multiplier=band_multiplier)
    future_states = states_all[states_all["date"].isin(forecast_series.index)].copy()

    # Output section: visualization
    st.subheader("2) Standardized Fusion Result and Cycle Phases")

    col1, col2 = st.columns([2, 1])
    with col1:
        phase_fig = make_phase_plot(states_df)
        st.plotly_chart(phase_fig, use_container_width=True)
    with col2:
        st.write("Fusion weights (based on chosen method):")
        wdf = pd.DataFrame({"indicator": list(weights.keys()), "weight": list(weights.values())}).sort_values(
            "weight", ascending=False
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        # Last period indicator contribution proxy
        X_use = df_tx[indicator_cols].copy()
        inv = set([c for c in config.inverse_columns])
        for c in X_use.columns:
            if c in inv:
                X_use[c] = -X_use[c]
        if config.fusion_method == "entropy":
            X_for_contrib = standardize_matrix(X_use, method="minmax")
        else:
            X_for_contrib = standardize_matrix(X_use, method=config.standardize)
        last_std = X_for_contrib.iloc[-1]
        contrib_fig = make_indicator_contrib_plot(last_std, weights=weights)
        st.plotly_chart(contrib_fig, use_container_width=True)

    st.subheader("3) Phase Timeline (State Segments)")
    # Segment by state changes
    timeline = []
    s_prev = None
    seg_start = None
    for _, row in states_df.iterrows():
        s = row["state"]
        if s_prev is None:
            s_prev = s
            seg_start = row["date"]
            continue
        if s != s_prev:
            timeline.append((seg_start, row["date"], s_prev))
            seg_start = row["date"]
            s_prev = s
    if s_prev is not None:
        timeline.append((seg_start, states_df["date"].iloc[-1], s_prev))
    tdf = pd.DataFrame(timeline, columns=["start", "end", "state"])
    tdf["state_cn"] = tdf["state"].map(
        {"Expansion": "扩张", "Peak": "滞涨/高点", "Contraction": "收缩", "Trough": "低谷"}
    )
    tdf = tdf.sort_values("start")
    st.dataframe(tdf[["start", "end", "state_cn"]], use_container_width=True, hide_index=True)

    st.subheader("4) Forecast (3-6 months) and Future Phase")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=composite.index, y=composite.values, mode="lines", name="Historical Composite"))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode="lines+markers", name="Forecast"))

    fig.update_layout(
        title=f"Composite Index Forecast ({horizon_months} months)",
        xaxis_title="Date",
        yaxis_title="Composite",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    future_states = future_states.sort_values("date")
    future_table = future_states[["date", "state", "state_cn", "composite"]].copy()
    future_table = future_table.rename(columns={"composite": "forecast_composite"})
    st.dataframe(future_table, use_container_width=True, hide_index=True)

    st.info(
        "Notes: This prototype uses generic fusion + heuristic phase classification and Holt-Winters forecasting. "
        "To match your thesis requirements, replace `create_sample_data()` with your real macro dataset "
        "and adjust transform/inversion choices for each indicator."
    )


if __name__ == "__main__":
    main()

