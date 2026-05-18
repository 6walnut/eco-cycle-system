"""
Fetch macro indicators for analysis from publicly available online sources.

Primary implementation uses AKShare macro interfaces, which aggregate public
macro series (including Sina-related datasets in its upstream sources).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

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
    # First parse Chinese month labels; then fallback to common explicit formats.
    parsed = _parse_cn_month_series(x["date"])
    fallback = pd.to_datetime(x["date"], format="%Y-%m-%d", errors="coerce")
    fallback = fallback.fillna(pd.to_datetime(x["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"))
    fallback = fallback.fillna(pd.to_datetime(x["date"], format="%Y-%m", errors="coerce"))
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


def _try_fetch_series_from_ak(
    ak,
    func_names: List[str],
    out_col: str,
    date_candidates: List[str],
    value_candidates: List[str],
    value_keywords: List[str],
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    """
    Try multiple AKShare function names for one indicator.
    Returns (normalized_df or None, meta_detail).
    """
    attempts = []
    for fn in func_names:
        if not hasattr(ak, fn):
            attempts.append({"func": fn, "ok": False, "reason": "func_not_found"})
            continue
        try:
            raw_df = getattr(ak, fn)()
            dcol = _first_existing(raw_df.columns, date_candidates)
            vcol = _first_existing(raw_df.columns, value_candidates)
            if vcol is None:
                vcol = _pick_by_keywords(raw_df.columns, value_keywords)
            if dcol and vcol:
                out_df = _normalize_monthly(raw_df, dcol, vcol, out_col)
                return out_df, {
                    "ok": True,
                    "source_func": fn,
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(out_df, out_col),
                    "attempts": attempts,
                }
            attempts.append(
                {
                    "func": fn,
                    "ok": False,
                    "reason": "column_match_failed",
                    "columns": [str(c) for c in raw_df.columns],
                }
            )
        except Exception as e:
            attempts.append({"func": fn, "ok": False, "reason": "fetch_exception", "error": str(e)})
    return None, {"ok": False, "reason": "all_candidates_failed", "attempts": attempts}


def _fetch_eastmoney_series(
    report_name: str,
    columns: str,
    value_field: str,
    out_col: str,
    date_field: str = "TIME",
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    """
    Web-direct fallback: fetch one macro series from Eastmoney datacenter API.
    """
    base = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "columns": columns,
        "pageNumber": "1",
        "pageSize": "2000",
        "sortColumns": "REPORT_DATE",
        "sortTypes": "-1",
        "source": "WEB",
        "client": "WEB",
        "reportName": report_name,
        "p": "1",
        "pageNo": "1",
        "pageNum": "1",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://data.eastmoney.com/",
        },
    )
    try:
        raw = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "ignore")
        data_json = json.loads(raw)
        rows = ((data_json or {}).get("result") or {}).get("data") or []
        if not rows:
            return None, {"ok": False, "reason": "empty_result", "channel": "web_fallback", "report_name": report_name}
        raw_df = pd.DataFrame(rows)
        if date_field not in raw_df.columns or value_field not in raw_df.columns:
            return None, {
                "ok": False,
                "reason": "column_match_failed",
                "channel": "web_fallback",
                "report_name": report_name,
                "columns": [str(c) for c in raw_df.columns],
                "required_date_field": date_field,
                "required_value_field": value_field,
            }
        out_df = _normalize_monthly(raw_df, date_field, value_field, out_col)
        return out_df, {
            "ok": True,
            "channel": "web_fallback",
            "source": "eastmoney_datacenter",
            "report_name": report_name,
            "date_col": date_field,
            "value_col": value_field,
            **_series_range_info(out_df, out_col),
        }
    except Exception as e:
        return None, {
            "ok": False,
            "reason": "fetch_exception",
            "channel": "web_fallback",
            "report_name": report_name,
            "error": str(e),
        }


def _fetch_social_finance_yoy_web() -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    """
    Web-direct fallback for social financing:
    fetch increment data and derive YoY (12-month pct change).
    """
    url = "https://data.mofcom.gov.cn/datamofcom/front/gnmy/shrzgmQuery"
    req = urllib.request.Request(
        url,
        data=b"",
        method="POST",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://data.mofcom.gov.cn/gnmy/shrzgm.shtml",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    try:
        raw = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "ignore")
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows:
            return None, {"ok": False, "reason": "empty_result", "channel": "web_fallback"}
        raw_df = pd.DataFrame(rows)
        if raw_df.empty:
            return None, {"ok": False, "reason": "empty_result", "channel": "web_fallback"}
        # Keep same ordering assumption as AKShare implementation.
        if len(raw_df.columns) >= 7:
            raw_df = raw_df.copy()
            raw_df.columns = [
                "月份",
                "其中-未贴现银行承兑汇票",
                "其中-委托贷款",
                "其中-委托贷款外币贷款",
                "其中-人民币贷款",
                "其中-企业债券",
                "社会融资规模增量",
                "其中-非金融企业境内股票融资",
                "其中-信托贷款",
            ][: len(raw_df.columns)]
        dcol = _first_existing(raw_df.columns, ["月份", "日期", "date"])
        vcol = _first_existing(raw_df.columns, ["社会融资规模增量"])
        if vcol is None:
            vcol = _pick_by_keywords(raw_df.columns, ["社会融资", "融资规模", "增量"])
        if dcol is None or vcol is None:
            return None, {
                "ok": False,
                "reason": "column_match_failed",
                "channel": "web_fallback",
                "columns": [str(c) for c in raw_df.columns],
            }
        amount_df = _normalize_monthly(raw_df, dcol, vcol, "social_finance_amount")
        amount_df["social_finance_yoy"] = amount_df["social_finance_amount"].pct_change(12) * 100.0
        out_df = amount_df[["date", "social_finance_yoy"]]
        return out_df, {
            "ok": True,
            "channel": "web_fallback",
            "source": "mofcom_shrzgm",
            "date_col": dcol,
            "value_col": vcol,
            "derived": "yoy_from_12m_pct_change",
            **_series_range_info(out_df, "social_finance_yoy"),
        }
    except Exception as e:
        return None, {"ok": False, "reason": "fetch_exception", "channel": "web_fallback", "error": str(e)}


def fetch_sina_macro_dataset_with_meta(
    enabled_indicators: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
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

    indicator_switch = enabled_indicators or {
        "cpi_yoy": True,
        "pmi": True,
        "m2_yoy": True,
        "m1_yoy": True,
        "ind_growth_yoy": True,
        "fai_acc_yoy": True,
        # Social financing is intentionally dropped from auto-crawl target
        # because upstream stability is poor in current deployment.
        "social_finance_yoy": False,
    }
    # Force disable social financing in crawl mode for now.
    indicator_switch["social_finance_yoy"] = False

    # 1) CPI
    if indicator_switch.get("cpi_yoy", True):
        try:
            cpi = ak.macro_china_cpi()
            dcol = _first_existing(cpi.columns, ["月份", "日期", "date"])
            vcol = _first_existing(cpi.columns, ["全国-同比增长", "同比增长", "全国-当月"])
            if vcol is None:
                vcol = _pick_by_keywords(cpi.columns, ["同比", "增长", "全国"])
            if dcol and vcol:
                cpi_df = _normalize_monthly(cpi, dcol, vcol, "cpi_yoy")
                merged = cpi_df if merged is None else merged.merge(cpi_df, on="date", how="outer")
                sources["cpi_yoy"] = {
                    "ok": True,
                    "channel": "akshare",
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(cpi_df, "cpi_yoy"),
                }
            else:
                cpi_df, cpi_meta = _fetch_eastmoney_series(
                    report_name="RPT_ECONOMY_CPI",
                    columns="REPORT_DATE,TIME,NATIONAL_SAME,NATIONAL_BASE,NATIONAL_SEQUENTIAL,NATIONAL_ACCUMULATE",
                    value_field="NATIONAL_SAME",
                    out_col="cpi_yoy",
                )
                if cpi_df is not None:
                    merged = cpi_df if merged is None else merged.merge(cpi_df, on="date", how="outer")
                    sources["cpi_yoy"] = cpi_meta
                else:
                    sources["cpi_yoy"] = {
                        "ok": False,
                        "reason": "column_match_failed",
                        "channel": "akshare_then_web",
                        "ak_columns": [str(c) for c in cpi.columns],
                        "web_detail": cpi_meta,
                    }
        except Exception as e:
            cpi_df, cpi_meta = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_CPI",
                columns="REPORT_DATE,TIME,NATIONAL_SAME,NATIONAL_BASE,NATIONAL_SEQUENTIAL,NATIONAL_ACCUMULATE",
                value_field="NATIONAL_SAME",
                out_col="cpi_yoy",
            )
            if cpi_df is not None:
                merged = cpi_df if merged is None else merged.merge(cpi_df, on="date", how="outer")
                sources["cpi_yoy"] = cpi_meta
            else:
                sources["cpi_yoy"] = {
                    "ok": False,
                    "reason": "fetch_exception",
                    "channel": "akshare_then_web",
                    "ak_error": str(e),
                    "web_detail": cpi_meta,
                }
    else:
        sources["cpi_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    # 2) PMI
    if indicator_switch.get("pmi", True):
        try:
            pmi = ak.macro_china_pmi()
            dcol = _first_existing(pmi.columns, ["月份", "日期", "date"])
            vcol = _pick_by_keywords(pmi.columns, ["制造业", "pmi", "指数"])
            if dcol and vcol:
                pmi_df = _normalize_monthly(pmi, dcol, vcol, "pmi")
                merged = pmi_df if merged is None else merged.merge(pmi_df, on="date", how="outer")
                sources["pmi"] = {
                    "ok": True,
                    "channel": "akshare",
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(pmi_df, "pmi"),
                }
            else:
                pmi_df, pmi_meta = _fetch_eastmoney_series(
                    report_name="RPT_ECONOMY_PMI",
                    columns="REPORT_DATE,TIME,MAKE_INDEX,MAKE_SAME,NMAKE_INDEX,NMAKE_SAME",
                    value_field="MAKE_INDEX",
                    out_col="pmi",
                )
                if pmi_df is not None:
                    merged = pmi_df if merged is None else merged.merge(pmi_df, on="date", how="outer")
                    sources["pmi"] = pmi_meta
                else:
                    sources["pmi"] = {
                        "ok": False,
                        "reason": "column_match_failed",
                        "channel": "akshare_then_web",
                        "ak_columns": [str(c) for c in pmi.columns],
                        "web_detail": pmi_meta,
                    }
        except Exception as e:
            pmi_df, pmi_meta = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_PMI",
                columns="REPORT_DATE,TIME,MAKE_INDEX,MAKE_SAME,NMAKE_INDEX,NMAKE_SAME",
                value_field="MAKE_INDEX",
                out_col="pmi",
            )
            if pmi_df is not None:
                merged = pmi_df if merged is None else merged.merge(pmi_df, on="date", how="outer")
                sources["pmi"] = pmi_meta
            else:
                sources["pmi"] = {
                    "ok": False,
                    "reason": "fetch_exception",
                    "channel": "akshare_then_web",
                    "ak_error": str(e),
                    "web_detail": pmi_meta,
                }
    else:
        sources["pmi"] = {"ok": False, "reason": "disabled_by_admin"}

    # 3) M2
    money_supply_df = None
    if indicator_switch.get("m2_yoy", True):
        try:
            money_supply_df = ak.macro_china_money_supply()
            dcol = _first_existing(money_supply_df.columns, ["月份", "日期", "date"])
            # Prefer M2 yoy column; fallback to keywords.
            vcol = _first_existing(money_supply_df.columns, ["货币和准货币(M2)-同比增长", "M2-同比增长"])
            if vcol is None:
                vcol = _pick_by_keywords(money_supply_df.columns, ["M2", "同比", "增长"])
            if dcol and vcol:
                m2_df = _normalize_monthly(money_supply_df, dcol, vcol, "m2_yoy")
                merged = m2_df if merged is None else merged.merge(m2_df, on="date", how="outer")
                sources["m2_yoy"] = {
                    "ok": True,
                    "channel": "akshare",
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(m2_df, "m2_yoy"),
                }
            else:
                m2_df, m2_meta = _fetch_eastmoney_series(
                    report_name="RPT_ECONOMY_CURRENCY_SUPPLY",
                    columns="REPORT_DATE,TIME,BASIC_CURRENCY,BASIC_CURRENCY_SAME,BASIC_CURRENCY_SEQUENTIAL,CURRENCY,CURRENCY_SAME,CURRENCY_SEQUENTIAL,FREE_CASH,FREE_CASH_SAME,FREE_CASH_SEQUENTIAL",
                    value_field="BASIC_CURRENCY_SAME",
                    out_col="m2_yoy",
                )
                if m2_df is not None:
                    merged = m2_df if merged is None else merged.merge(m2_df, on="date", how="outer")
                    sources["m2_yoy"] = m2_meta
                else:
                    sources["m2_yoy"] = {
                        "ok": False,
                        "reason": "column_match_failed",
                        "channel": "akshare_then_web",
                        "ak_columns": [str(c) for c in money_supply_df.columns],
                        "web_detail": m2_meta,
                    }
        except Exception as e:
            m2_df, m2_meta = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_CURRENCY_SUPPLY",
                columns="REPORT_DATE,TIME,BASIC_CURRENCY,BASIC_CURRENCY_SAME,BASIC_CURRENCY_SEQUENTIAL,CURRENCY,CURRENCY_SAME,CURRENCY_SEQUENTIAL,FREE_CASH,FREE_CASH_SAME,FREE_CASH_SEQUENTIAL",
                value_field="BASIC_CURRENCY_SAME",
                out_col="m2_yoy",
            )
            if m2_df is not None:
                merged = m2_df if merged is None else merged.merge(m2_df, on="date", how="outer")
                sources["m2_yoy"] = m2_meta
            else:
                sources["m2_yoy"] = {
                    "ok": False,
                    "reason": "fetch_exception",
                    "channel": "akshare_then_web",
                    "ak_error": str(e),
                    "web_detail": m2_meta,
                }
    else:
        sources["m2_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    # 4) M1
    if indicator_switch.get("m1_yoy", True):
        try:
            if money_supply_df is None:
                money_supply_df = ak.macro_china_money_supply()
            dcol = _first_existing(money_supply_df.columns, ["月份", "日期", "date"])
            # Prefer M1 yoy column; fallback to keywords.
            vcol = _first_existing(money_supply_df.columns, ["货币(M1)-同比增长", "M1-同比增长"])
            if vcol is None:
                vcol = _pick_by_keywords(money_supply_df.columns, ["M1", "同比", "增长"])
            if dcol and vcol:
                m1_df = _normalize_monthly(money_supply_df, dcol, vcol, "m1_yoy")
                merged = m1_df if merged is None else merged.merge(m1_df, on="date", how="outer")
                sources["m1_yoy"] = {
                    "ok": True,
                    "channel": "akshare",
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(m1_df, "m1_yoy"),
                }
            else:
                m1_df, m1_meta = _fetch_eastmoney_series(
                    report_name="RPT_ECONOMY_CURRENCY_SUPPLY",
                    columns="REPORT_DATE,TIME,BASIC_CURRENCY,BASIC_CURRENCY_SAME,BASIC_CURRENCY_SEQUENTIAL,CURRENCY,CURRENCY_SAME,CURRENCY_SEQUENTIAL,FREE_CASH,FREE_CASH_SAME,FREE_CASH_SEQUENTIAL",
                    value_field="CURRENCY_SAME",
                    out_col="m1_yoy",
                )
                if m1_df is not None:
                    merged = m1_df if merged is None else merged.merge(m1_df, on="date", how="outer")
                    sources["m1_yoy"] = m1_meta
                else:
                    sources["m1_yoy"] = {
                        "ok": False,
                        "reason": "column_match_failed",
                        "channel": "akshare_then_web",
                        "ak_columns": [str(c) for c in money_supply_df.columns],
                        "web_detail": m1_meta,
                    }
        except Exception as e:
            m1_df, m1_meta = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_CURRENCY_SUPPLY",
                columns="REPORT_DATE,TIME,BASIC_CURRENCY,BASIC_CURRENCY_SAME,BASIC_CURRENCY_SEQUENTIAL,CURRENCY,CURRENCY_SAME,CURRENCY_SEQUENTIAL,FREE_CASH,FREE_CASH_SAME,FREE_CASH_SEQUENTIAL",
                value_field="CURRENCY_SAME",
                out_col="m1_yoy",
            )
            if m1_df is not None:
                merged = m1_df if merged is None else merged.merge(m1_df, on="date", how="outer")
                sources["m1_yoy"] = m1_meta
            else:
                sources["m1_yoy"] = {
                    "ok": False,
                    "reason": "fetch_exception",
                    "channel": "akshare_then_web",
                    "ak_error": str(e),
                    "web_detail": m1_meta,
                }
    else:
        sources["m1_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    # 5) Social financing
    if indicator_switch.get("social_finance_yoy", True):
        try:
            sf = ak.macro_china_shrzgm()
            dcol = _first_existing(sf.columns, ["月份", "日期", "date"])
            vcol = _pick_by_keywords(sf.columns, ["社融", "融资", "同比", "存量"])
            if dcol and vcol:
                sf_df = _normalize_monthly(sf, dcol, vcol, "social_finance_yoy")
                merged = sf_df if merged is None else merged.merge(sf_df, on="date", how="outer")
                sources["social_finance_yoy"] = {
                    "ok": True,
                    "channel": "akshare",
                    "date_col": dcol,
                    "value_col": vcol,
                    **_series_range_info(sf_df, "social_finance_yoy"),
                }
            else:
                sf_df, sf_meta = _fetch_social_finance_yoy_web()
                if sf_df is not None:
                    merged = sf_df if merged is None else merged.merge(sf_df, on="date", how="outer")
                    sources["social_finance_yoy"] = sf_meta
                else:
                    sources["social_finance_yoy"] = {
                        "ok": False,
                        "reason": "column_match_failed",
                        "channel": "akshare_then_web",
                        "ak_columns": [str(c) for c in sf.columns],
                        "web_detail": sf_meta,
                    }
        except Exception as e:
            sf_df, sf_meta = _fetch_social_finance_yoy_web()
            if sf_df is not None:
                merged = sf_df if merged is None else merged.merge(sf_df, on="date", how="outer")
                sources["social_finance_yoy"] = sf_meta
            else:
                sources["social_finance_yoy"] = {
                    "ok": False,
                    "reason": "fetch_exception",
                    "channel": "akshare_then_web",
                    "ak_error": str(e),
                    "web_detail": sf_meta,
                }
    else:
        sources["social_finance_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    # 6) Industrial value-added growth yoy
    if indicator_switch.get("ind_growth_yoy", True):
        ind_df, ind_meta = _try_fetch_series_from_ak(
            ak=ak,
            func_names=[
                "macro_china_industrial_production",
                "macro_china_industrial_added_value",
                "macro_china_industrial_increase",
                "macro_china_industrial_production_yoy",
            ],
            out_col="ind_growth_yoy",
            date_candidates=["月份", "日期", "date", "时间"],
            value_candidates=[
                "工业增加值-同比增长",
                "规模以上工业增加值-同比增长",
                "当月同比",
                "同比增长",
            ],
            value_keywords=["工业", "增加值", "同比", "增长"],
        )
        if ind_df is not None:
            merged = ind_df if merged is None else merged.merge(ind_df, on="date", how="outer")
            ind_meta["channel"] = "akshare"
            sources["ind_growth_yoy"] = ind_meta
        else:
            ind_df_web, ind_meta_web = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_INDUS_GROW",
                columns="REPORT_DATE,TIME,BASE_SAME,BASE_ACCUMULATE",
                value_field="BASE_SAME",
                out_col="ind_growth_yoy",
            )
            if ind_df_web is not None:
                merged = ind_df_web if merged is None else merged.merge(ind_df_web, on="date", how="outer")
                sources["ind_growth_yoy"] = ind_meta_web
            else:
                sources["ind_growth_yoy"] = {
                    "ok": False,
                    "reason": "all_candidates_failed",
                    "channel": "akshare_then_web",
                    "ak_detail": ind_meta,
                    "web_detail": ind_meta_web,
                }
    else:
        sources["ind_growth_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    # 7) Fixed asset investment cumulative yoy
    if indicator_switch.get("fai_acc_yoy", True):
        fai_df, fai_meta = _try_fetch_series_from_ak(
            ak=ak,
            func_names=[
                "macro_china_fixed_asset_investment",
                "macro_china_fai",
                "macro_china_investment_in_fixed_assets",
            ],
            out_col="fai_acc_yoy",
            date_candidates=["月份", "日期", "date", "时间"],
            value_candidates=[
                "固定资产投资完成额-累计同比",
                "固定资产投资累计同比",
                "累计同比增长",
                "累计同比",
            ],
            value_keywords=["固定资产", "投资", "累计", "同比"],
        )
        if fai_df is not None:
            merged = fai_df if merged is None else merged.merge(fai_df, on="date", how="outer")
            fai_meta["channel"] = "akshare"
            sources["fai_acc_yoy"] = fai_meta
        else:
            fai_df_web, fai_meta_web = _fetch_eastmoney_series(
                report_name="RPT_ECONOMY_ASSET_INVEST",
                columns="REPORT_DATE,TIME,BASE,BASE_SAME,BASE_SEQUENTIAL,BASE_ACCUMULATE",
                value_field="BASE_SAME",
                out_col="fai_acc_yoy",
            )
            if fai_df_web is not None:
                merged = fai_df_web if merged is None else merged.merge(fai_df_web, on="date", how="outer")
                sources["fai_acc_yoy"] = fai_meta_web
            else:
                sources["fai_acc_yoy"] = {
                    "ok": False,
                    "reason": "all_candidates_failed",
                    "channel": "akshare_then_web",
                    "ak_detail": fai_meta,
                    "web_detail": fai_meta_web,
                }
    else:
        sources["fai_acc_yoy"] = {"ok": False, "reason": "disabled_by_admin"}

    if merged is None or "date" not in merged.columns:
        raise RuntimeError(f"Unable to fetch macro series from online source currently. details={sources}")

    merged = merged.sort_values("date").reset_index(drop=True)
    # Auto-crawl mode currently targets 6 stable indicators.
    expected_cols = ["cpi_yoy", "pmi", "m2_yoy", "m1_yoy", "ind_growth_yoy", "fai_acc_yoy"]
    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[["date"] + expected_cols]

    # Keep rows with at least one indicator value; downstream logic will interpolate/fill.
    merged = merged.dropna(subset=expected_cols, how="all")
    available_cols = [c for c in expected_cols if merged[c].notna().any()]
    if len(available_cols) < 2:
        raise RuntimeError(f"Fetched indicators are insufficient (<2 non-null columns). details={sources}")
    non_null_counts = {c: int(merged[c].notna().sum()) for c in expected_cols}
    meta = {
        "sources": sources,
        "merged_rows": int(len(merged)),
        "date_min": str(merged["date"].min().date()) if not merged.empty else None,
        "date_max": str(merged["date"].max().date()) if not merged.empty else None,
        "non_null_counts": non_null_counts,
        "expected_indicators": expected_cols,
        "available_indicators": available_cols,
    }
    return merged, meta


def fetch_sina_macro_dataset() -> pd.DataFrame:
    df, _meta = fetch_sina_macro_dataset_with_meta()
    return df

