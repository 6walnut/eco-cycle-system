"""
Fetch macro indicators for analysis from publicly available online sources.

Primary implementation uses AKShare macro interfaces, which aggregate public
macro series (including Sina-related datasets in its upstream sources).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cset = list(cols)
    for name in candidates:
        if name in cset:
            return name
    return None


def _pick_by_keywords(cols: Iterable[str], include_any: Iterable[str]) -> Optional[str]:
    inc = list(include_any)
    for c in cols:
        lc = str(c).lower()
        if any(k.lower() in lc for k in inc):
            return c
    return None


def _parse_cn_month_series(s: pd.Series) -> pd.Series:
    """
    Parse month strings like '2026年02月份' into month-start timestamps.
    """
    x = s.astype(str).str.strip()
    x = x.str.replace("年", "-", regex=False)
    x = x.str.replace("月份", "", regex=False)
    x = x.str.replace("月", "", regex=False)
    # Keep only year-month pattern; invalid rows become NaN.
    ym = x.str.extract(r"(?P<y>\d{4})-(?P<m>\d{1,2})")
    y = pd.to_numeric(ym["y"], errors="coerce")
    m = pd.to_numeric(ym["m"], errors="coerce")
    valid = y.notna() & m.notna()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[valid] = pd.to_datetime(
        y.loc[valid].astype(int).astype(str) + "-" + m.loc[valid].astype(int).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return out


def _normalize_monthly(df: pd.DataFrame, date_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    x = df[[date_col, value_col]].copy()
    x.columns = ["date", out_col]
    # First parse Chinese month labels; then fallback to pandas generic parser.
    parsed = _parse_cn_month_series(x["date"])
    fallback = pd.to_datetime(x["date"], errors="coerce")
    x["date"] = parsed.fillna(fallback)
    x = x.dropna(subset=["date"])
    x["date"] = x["date"].dt.to_period("M").dt.to_timestamp(how="start")
    raw = x[out_col].astype(str).str.strip()
    raw = raw.str.replace(",", "", regex=False)
    raw = raw.str.replace("%", "", regex=False)
    raw = raw.str.replace("％", "", regex=False)
    # Keep only numeric-like chars to handle units/symbols from upstream sources.
    raw = raw.str.replace(r"[^0-9eE\.\+\-]", "", regex=True)
    x[out_col] = pd.to_numeric(raw, errors="coerce")
    return x.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _series_range_info(df: pd.DataFrame, value_col: str) -> Dict[str, object]:
    if df.empty:
        return {"rows": 0, "non_null": 0, "date_min": None, "date_max": None}
    nn = int(df[value_col].notna().sum())
    if nn == 0:
        return {"rows": int(len(df)), "non_null": 0, "date_min": None, "date_max": None}
    non_null_df = df[df[value_col].notna()]
    return {
        "rows": int(len(df)),
        "non_null": nn,
        "date_min": str(non_null_df["date"].min().date()),
        "date_max": str(non_null_df["date"].max().date()),
    }


def fetch_sina_macro_dataset_with_meta() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build a monthly macro dataset for direct analysis without manual CSV upload.
    Returns at least: date + >=2 indicators.
    """
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("Missing dependency `akshare`. Install with: pip install akshare") from e

    merged = None
    sources: Dict[str, Dict[str, object]] = {}

    # 1) CPI
    try:
        cpi = ak.macro_china_cpi()
        dcol = _first_existing(cpi.columns, ["月份", "日期", "date"])
        vcol = _first_existing(cpi.columns, ["全国-同比增长", "同比增长", "全国-当月"])
        if vcol is None:
            vcol = _pick_by_keywords(cpi.columns, ["同比", "增长", "全国"])
        if dcol and vcol:
            cpi_df = _normalize_monthly(cpi, dcol, vcol, "cpi_yoy")
            merged = cpi_df if merged is None else merged.merge(cpi_df, on="date", how="outer")
            sources["cpi_yoy"] = {"ok": True, "date_col": dcol, "value_col": vcol, **_series_range_info(cpi_df, "cpi_yoy")}
        else:
            sources["cpi_yoy"] = {"ok": False, "reason": "column_match_failed", "columns": [str(c) for c in cpi.columns]}
    except Exception:
        sources["cpi_yoy"] = {"ok": False, "reason": "fetch_exception"}

    # 2) PMI
    try:
        pmi = ak.macro_china_pmi()
        dcol = _first_existing(pmi.columns, ["月份", "日期", "date"])
        vcol = _pick_by_keywords(pmi.columns, ["制造业", "pmi", "指数"])
        if dcol and vcol:
            pmi_df = _normalize_monthly(pmi, dcol, vcol, "pmi")
            merged = pmi_df if merged is None else merged.merge(pmi_df, on="date", how="outer")
            sources["pmi"] = {"ok": True, "date_col": dcol, "value_col": vcol, **_series_range_info(pmi_df, "pmi")}
        else:
            sources["pmi"] = {"ok": False, "reason": "column_match_failed", "columns": [str(c) for c in pmi.columns]}
    except Exception:
        sources["pmi"] = {"ok": False, "reason": "fetch_exception"}

    # 3) M2
    try:
        m2 = ak.macro_china_money_supply()
        dcol = _first_existing(m2.columns, ["月份", "日期", "date"])
        # Prefer M2 yoy column; fallback to keywords.
        vcol = _first_existing(m2.columns, ["货币和准货币(M2)-同比增长", "M2-同比增长"])
        if vcol is None:
            vcol = _pick_by_keywords(m2.columns, ["M2", "同比", "增长"])
        if dcol and vcol:
            m2_df = _normalize_monthly(m2, dcol, vcol, "m2_yoy")
            merged = m2_df if merged is None else merged.merge(m2_df, on="date", how="outer")
            sources["m2_yoy"] = {"ok": True, "date_col": dcol, "value_col": vcol, **_series_range_info(m2_df, "m2_yoy")}
        else:
            sources["m2_yoy"] = {"ok": False, "reason": "column_match_failed", "columns": [str(c) for c in m2.columns]}
    except Exception:
        sources["m2_yoy"] = {"ok": False, "reason": "fetch_exception"}

    # 4) Social financing
    try:
        sf = ak.macro_china_shrzgm()
        dcol = _first_existing(sf.columns, ["月份", "日期", "date"])
        vcol = _pick_by_keywords(sf.columns, ["社融", "融资", "同比", "存量"])
        if dcol and vcol:
            sf_df = _normalize_monthly(sf, dcol, vcol, "social_finance_yoy")
            merged = sf_df if merged is None else merged.merge(sf_df, on="date", how="outer")
            sources["social_finance_yoy"] = {
                "ok": True,
                "date_col": dcol,
                "value_col": vcol,
                **_series_range_info(sf_df, "social_finance_yoy"),
            }
        else:
            sources["social_finance_yoy"] = {
                "ok": False,
                "reason": "column_match_failed",
                "columns": [str(c) for c in sf.columns],
            }
    except Exception:
        sources["social_finance_yoy"] = {"ok": False, "reason": "fetch_exception"}

    if merged is None or "date" not in merged.columns:
        raise RuntimeError(f"Unable to fetch macro series from online source currently. details={sources}")

    merged = merged.sort_values("date").reset_index(drop=True)
    # Keep rows with at least two indicator values to ensure model input quality.
    indicator_cols = [c for c in merged.columns if c != "date"]
    merged = merged.dropna(subset=indicator_cols, how="all")
    if len(indicator_cols) < 2:
        raise RuntimeError(f"Fetched indicators are insufficient (<2 columns). details={sources}")
    non_null_counts = {c: int(merged[c].notna().sum()) for c in indicator_cols}
    meta = {
        "sources": sources,
        "merged_rows": int(len(merged)),
        "date_min": str(merged["date"].min().date()) if not merged.empty else None,
        "date_max": str(merged["date"].max().date()) if not merged.empty else None,
        "non_null_counts": non_null_counts,
    }
    return merged, meta


def fetch_sina_macro_dataset() -> pd.DataFrame:
    df, _meta = fetch_sina_macro_dataset_with_meta()
    return df

