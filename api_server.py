import csv
import io
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from core import FusionConfig, _parse_date_col, create_sample_data, load_user_csv, run_analysis
from db_models import (
    add_audit_log,
    add_favorite_run,
    assign_dataset_owner,
    assign_run_owner,
    create_share_link,
    create_user,
    delete_analysis_run,
    delete_model,
    delete_user,
    get_analysis_run,
    get_share_link,
    get_system_config,
    get_user,
    get_user_by_username,
    get_user_run_stats,
    init_db,
    list_datasets,
    list_favorite_runs,
    list_models,
    list_audit_logs,
    list_system_configs,
    list_user_datasets,
    list_user_runs,
    list_users,
    load_dataset_rows,
    remove_favorite_run,
    save_analysis_run,
    save_dataset,
    set_system_config,
    upsert_model,
    user_owns_dataset,
    user_owns_run,
)
from sina_macro_fetch import fetch_sina_macro_dataset_with_meta

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)

_SINA_CACHE: Dict[str, object] = {"ts": None, "df": None, "meta": None}


def _valid_username(username: str) -> bool:
    # Allow Chinese, letters, digits and underscore, 3-32 chars, no spaces.
    return bool(re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff]{3,32}", username))


def _parse_inverse_columns(raw: Optional[str], all_indicators: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return []
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return [c for c in cols if c in all_indicators]


def _token_serializer(app: Flask) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="auth-token")


def _make_token(app: Flask, user_id: int, username: str, is_admin: bool) -> str:
    ser = _token_serializer(app)
    return ser.dumps({"uid": user_id, "u": username, "admin": bool(is_admin)})


def _read_token(app: Flask, token: str, max_age_seconds: int = 7 * 24 * 3600) -> Optional[dict]:
    ser = _token_serializer(app)
    try:
        return ser.loads(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None


def _auth_user(app: Flask, required: bool = False) -> Optional[dict]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        if required:
            return None
        return None
    token = auth.split(" ", 1)[1].strip()
    payload = _read_token(app, token)
    if payload is None:
        return None
    user = get_user(int(payload.get("uid", 0)))
    return user


def _ensure_admin(user: Optional[dict]) -> bool:
    return bool(user and user.get("is_admin"))


def _analyze_params_from_form():
    transform_type = request.form.get("transform_type", "none")
    standardize = request.form.get("standardize", "zscore")
    fusion_method = request.form.get("fusion_method", "pca")
    smoothing_window = int(request.form.get("smoothing_window", "3"))
    horizon_months = int(request.form.get("horizon_months", "3"))
    ma_window = int(request.form.get("ma_window", "6"))
    band_multiplier = float(request.form.get("band_multiplier", "0.3"))
    clip_lo = float(request.form.get("clip_lo", "0.01"))
    clip_hi = float(request.form.get("clip_hi", "0.99"))
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    forecast_model = request.form.get("forecast_model", "hw")  # hw | lstm
    if forecast_model not in {"hw", "lstm"}:
        forecast_model = "hw"
    fusion_method = fusion_method if fusion_method in {"equal", "pca", "entropy", "dfm"} else "pca"
    return {
        "transform_type": transform_type,
        "standardize": standardize,
        "fusion_method": fusion_method,
        "smoothing_window": smoothing_window,
        "horizon_months": horizon_months,
        "ma_window": ma_window,
        "band_multiplier": band_multiplier,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_model": forecast_model,
    }


def _run_analyze_df(df: pd.DataFrame, params: dict, inverse_columns: List[str]):
    indicator_cols = [c for c in df.columns if c != "date"]
    config = FusionConfig(
        transform_type=params["transform_type"],
        standardize=params["standardize"],
        fusion_method=params["fusion_method"],
        inverse_columns=inverse_columns,
        smoothing_window=params["smoothing_window"],
    )
    return run_analysis(
        df,
        config=config,
        clip_quantiles=(params["clip_lo"], params["clip_hi"]),
        horizon_months=params["horizon_months"],
        ma_window=params["ma_window"],
        band_multiplier=params["band_multiplier"],
        start_date=params["start_date"],
        end_date=params["end_date"],
        forecast_model=params["forecast_model"],
    )


def _rows_for_storage(df: pd.DataFrame) -> List[dict]:
    x = _parse_date_col(df.copy())
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    x = x.replace({np.nan: None, pd.NA: None})
    return x.to_dict(orient="records")


def _persist_run(
    df: pd.DataFrame, params: dict, result: dict, dataset_name: str = "upload", user_id: Optional[int] = None
) -> tuple[int, int]:
    rows = _rows_for_storage(df)
    ds_id = save_dataset(rows, name=dataset_name)
    run_id = save_analysis_run(ds_id, params, result, forecast_model=params["forecast_model"])
    if user_id is not None:
        assign_dataset_owner(ds_id, user_id)
        assign_run_owner(run_id, user_id)
    return ds_id, run_id


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("APP_SECRET_KEY", "dev-secret-change-me")
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_db()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/sample")
    def sample():
        df = create_sample_data()
        return jsonify({"csv_preview": df.head(10).to_dict(orient="records"), "rows": len(df)})

    @app.post("/api/auth/register")
    def auth_register():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", "")).strip()
        if (not _valid_username(username)) or len(password) < 6:
            return jsonify({"error": "用户名需为3-32位（中文/字母/数字/下划线），密码至少6位"}), 400
        try:
            uid = create_user(username, generate_password_hash(password), is_admin=False)
        except ValueError as e:
            msg = str(e)
            if "username already exists" in msg:
                msg = "用户名已存在，请更换。"
            return jsonify({"error": msg}), 400
        user = get_user(uid)
        add_audit_log("user_register", user_id=user["id"], username=user["username"], detail="self register")
        token = _make_token(app, user["id"], user["username"], user["is_admin"])
        return jsonify({"token": token, "user": user})

    @app.post("/api/auth/login")
    def auth_login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", "")).strip()
        role = str(payload.get("role", "user")).strip().lower()  # user | admin
        u = get_user_by_username(username)
        if u is None or not check_password_hash(u["password_hash"], password):
            return jsonify({"error": "用户名或密码错误"}), 401
        if role == "admin" and not bool(u["is_admin"]):
            return jsonify({"error": "该账号不是管理员，请使用用户登录。"}), 403
        if role == "user" and bool(u["is_admin"]):
            return jsonify({"error": "管理员账号请使用管理员登录入口。"}), 403
        add_audit_log("user_login", user_id=u["id"], username=u["username"], detail=f"role={role}")
        token = _make_token(app, u["id"], u["username"], u["is_admin"])
        return jsonify(
            {
                "token": token,
                "user": {"id": u["id"], "username": u["username"], "is_admin": bool(u["is_admin"])},
            }
        )

    @app.get("/api/me")
    def me():
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        return jsonify(user)

    @app.post("/api/analyze")
    def analyze():
        user = _auth_user(app, required=False)
        use_sample = str(request.form.get("use_sample", "false")).lower() in {"1", "true", "yes"}
        uploaded = request.files.get("file")
        params = _analyze_params_from_form()

        if use_sample:
            df = create_sample_data()
        else:
            if uploaded is None:
                return jsonify({"error": "缺少 CSV 文件：请上传 `file`，或设置 use_sample=true"}), 400
            df = load_user_csv(uploaded)

        indicator_cols = [c for c in df.columns if c != "date"]
        inverse_columns = _parse_inverse_columns(request.form.get("inverse_columns"), indicator_cols)

        try:
            result = _run_analyze_df(df, params, inverse_columns)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        # Auto-persist every analysis run
        try:
            dataset_name = uploaded.filename if (uploaded is not None and uploaded.filename) else "sample"
            ds_id, run_id = _persist_run(
                df, params, result, dataset_name=dataset_name, user_id=(user["id"] if user else None)
            )
            if user:
                add_audit_log("analyze_run", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
            result["dataset_id"] = ds_id
            result["run_id"] = run_id
        except Exception as e:
            return jsonify({"error": f"分析成功，但保存数据库失败：{e}"}), 500

        return jsonify(result)

    @app.post("/api/analyze/sina")
    def analyze_sina():
        """
        Auto-fetch macro data from online source and run analysis directly.
        No file upload required.
        """
        user = _auth_user(app, required=False)
        params = _analyze_params_from_form()
        enabled = get_system_config("akshare_indicators", {})
        try:
            interval_minutes = int(get_system_config("akshare_update_interval_minutes", 60))
        except Exception:
            interval_minutes = 60
        interval_minutes = max(5, interval_minutes)

        try:
            now = datetime.utcnow()
            cached_ts = _SINA_CACHE.get("ts")
            if cached_ts is not None and _SINA_CACHE.get("df") is not None:
                if (now - cached_ts).total_seconds() < interval_minutes * 60:
                    df = _SINA_CACHE["df"]
                    fetch_meta = dict(_SINA_CACHE.get("meta") or {})
                    fetch_meta["cache"] = "hit"
                else:
                    df, fetch_meta = fetch_sina_macro_dataset_with_meta(enabled_indicators=enabled)
                    _SINA_CACHE["ts"] = now
                    _SINA_CACHE["df"] = df
                    _SINA_CACHE["meta"] = fetch_meta
            else:
                df, fetch_meta = fetch_sina_macro_dataset_with_meta(enabled_indicators=enabled)
                _SINA_CACHE["ts"] = now
                _SINA_CACHE["df"] = df
                _SINA_CACHE["meta"] = fetch_meta
        except Exception as e:
            return jsonify({"error": f"抓取新浪宏观数据失败：{e}"}), 400

        indicator_cols = [c for c in df.columns if c != "date"]
        inverse_columns = _parse_inverse_columns(request.form.get("inverse_columns"), indicator_cols)
        try:
            result = _run_analyze_df(df, params, inverse_columns)
        except Exception as e:
            # Fallback: if date filter removes all rows, retry once without date range.
            if "No data after applying date range filter." in str(e) and (params.get("start_date") or params.get("end_date")):
                retry_params = dict(params)
                retry_params["start_date"] = None
                retry_params["end_date"] = None
                try:
                    result = _run_analyze_df(df, retry_params, inverse_columns)
                    result["warnings"] = [
                        "日期范围过滤后无数据，系统已自动去掉 start_date/end_date 重跑。"
                    ]
                    result["effective_params"] = retry_params
                except Exception as retry_e:
                    return jsonify({"error": str(retry_e), "fetch_meta": fetch_meta}), 400
            else:
                return jsonify({"error": str(e), "fetch_meta": fetch_meta}), 400

        try:
            ds_id, run_id = _persist_run(df, params, result, dataset_name="sina_auto", user_id=(user["id"] if user else None))
            if user:
                add_audit_log("analyze_sina_run", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
            result["dataset_id"] = ds_id
            result["run_id"] = run_id
            result["source"] = "sina_online"
            result["fetch_meta"] = fetch_meta
        except Exception as e:
            return jsonify({"error": f"分析成功，但保存数据库失败：{e}"}), 500
        return jsonify(result)

    @app.post("/api/datasets")
    def upload_dataset():
        """Upload CSV and store rows in DB. Returns dataset id."""
        user = _auth_user(app, required=False)
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "缺少上传文件"}), 400
        name = request.form.get("name", "upload")
        raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
        df = _parse_date_col(df)
        rows = _rows_for_storage(df)
        ds_id = save_dataset(rows, name=name)
        if user:
            assign_dataset_owner(ds_id, user["id"])
            add_audit_log("upload_dataset", user_id=user["id"], username=user["username"], detail=f"dataset_id={ds_id}")
        return jsonify({"id": ds_id, "name": name, "rows": len(rows)})

    @app.get("/api/datasets")
    def datasets_list():
        return jsonify(list_datasets())

    @app.get("/api/datasets/<int:ds_id>")
    def dataset_get(ds_id: int):
        try:
            rows = load_dataset_rows(ds_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        return jsonify({"id": ds_id, "preview": rows[:50], "total": len(rows)})

    @app.post("/api/datasets/<int:ds_id>/analyze")
    def dataset_analyze(ds_id: int):
        user = _auth_user(app, required=False)
        if user and (not user.get("is_admin")) and (not user_owns_dataset(user["id"], ds_id)):
            return jsonify({"error": "无权限访问该数据集"}), 403
        try:
            rows = load_dataset_rows(ds_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        df = pd.DataFrame(rows)
        params = _analyze_params_from_form()
        indicator_cols = [c for c in df.columns if c != "date"]
        inverse_columns = _parse_inverse_columns(request.form.get("inverse_columns"), indicator_cols)
        try:
            result = _run_analyze_df(df, params, inverse_columns)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        run_id = save_analysis_run(ds_id, params, result, forecast_model=params["forecast_model"])
        if user:
            assign_run_owner(run_id, user["id"])
            add_audit_log("rerun_dataset", user_id=user["id"], username=user["username"], detail=f"dataset_id={ds_id}, run_id={run_id}")
        result["dataset_id"] = ds_id
        result["run_id"] = run_id
        return jsonify(result)

    @app.get("/api/runs/<int:run_id>")
    def run_get(run_id: int):
        user = _auth_user(app, required=False)
        if user and (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限访问该分析结果"}), 403
        r = get_analysis_run(run_id)
        if r is None:
            return jsonify({"error": "未找到该记录"}), 404
        return jsonify(r)

    @app.delete("/api/runs/<int:run_id>")
    def run_delete(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限删除该分析记录"}), 403
        ok = delete_analysis_run(run_id)
        if not ok:
            return jsonify({"error": "分析记录不存在"}), 404
        add_audit_log("user_delete_run", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
        return jsonify({"ok": True})

    @app.get("/api/runs/compare")
    def compare_runs():
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        run_id_a = int(request.args.get("run_id_a", "0"))
        run_id_b = int(request.args.get("run_id_b", "0"))
        if run_id_a <= 0 or run_id_b <= 0:
            return jsonify({"error": "请提供 run_id_a 和 run_id_b"}), 400
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id_a) or not user_owns_run(user["id"], run_id_b)):
            return jsonify({"error": "无权限对比这两次分析"}), 403
        a = get_analysis_run(run_id_a)
        b = get_analysis_run(run_id_b)
        if a is None or b is None:
            return jsonify({"error": "分析记录不存在"}), 404
        wa = a["result"].get("weights", {})
        wb = b["result"].get("weights", {})
        keys = sorted(set(wa.keys()) | set(wb.keys()))
        weight_delta = {k: float(wb.get(k, 0.0)) - float(wa.get(k, 0.0)) for k in keys}
        return jsonify(
            {
                "run_a": {"id": run_id_a, "forecast_model": a.get("forecast_model"), "params": a.get("params")},
                "run_b": {"id": run_id_b, "forecast_model": b.get("forecast_model"), "params": b.get("params")},
                "weight_delta": weight_delta,
                "forecast_a": a["result"].get("forecast", []),
                "forecast_b": b["result"].get("forecast", []),
            }
        )

    @app.get("/api/me/datasets")
    def my_datasets():
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        return jsonify(list_user_datasets(user["id"]))

    @app.get("/api/me/runs")
    def my_runs():
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        return jsonify(list_user_runs(user["id"]))

    @app.get("/api/me/favorites")
    def my_favorites():
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        return jsonify(list_favorite_runs(user["id"]))

    @app.post("/api/favorites/<int:run_id>")
    def favorite_add(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限收藏该分析结果"}), 403
        add_favorite_run(user["id"], run_id)
        add_audit_log("favorite_add", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
        return jsonify({"ok": True})

    @app.delete("/api/favorites/<int:run_id>")
    def favorite_del(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        remove_favorite_run(user["id"], run_id)
        add_audit_log("favorite_remove", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
        return jsonify({"ok": True})

    @app.post("/api/runs/<int:run_id>/share")
    def share_run(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限分享该分析结果"}), 403
        expires_days = int((request.get_json(silent=True) or {}).get("expires_days", 7))
        token = secrets.token_urlsafe(24)
        expires_at = datetime.utcnow() + timedelta(days=max(1, min(expires_days, 365)))
        create_share_link(token, run_id, user["id"], expires_at)
        add_audit_log("share_create", user_id=user["id"], username=user["username"], detail=f"run_id={run_id}")
        api_base = request.host_url.rstrip("/")
        # Prefer browser origin for frontend deep-link sharing.
        app_base = (request.headers.get("Origin") or "").rstrip("/") or api_base
        return jsonify(
            {
                "share_url": f"{app_base}/system?shared_token={token}",
                "api_share_url": f"{api_base}/api/share/{token}",
                "token": token,
                "expires_at": expires_at.isoformat(),
            }
        )

    @app.get("/api/share/<token>")
    def share_get(token: str):
        row = get_share_link(token)
        if row is None:
            return jsonify({"error": "分享链接无效"}), 404
        if row["expires_at"] is not None and datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
            return jsonify({"error": "分享链接已过期"}), 410
        r = get_analysis_run(int(row["run_id"]))
        if r is None:
            return jsonify({"error": "分析记录不存在"}), 404
        return jsonify({"shared": True, "run": r})

    @app.get("/api/datasets/<int:ds_id>/download.csv")
    def dataset_download(ds_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_dataset(user["id"], ds_id)):
            return jsonify({"error": "无权限下载该数据集"}), 403
        rows = load_dataset_rows(ds_id)
        if not rows:
            return jsonify({"error": "数据集为空"}), 400
        fields = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="dataset_{ds_id}.csv"'},
        )

    @app.get("/api/runs/<int:run_id>/download.json")
    def run_download_json(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限下载该分析结果"}), 403
        r = get_analysis_run(run_id)
        if r is None:
            return jsonify({"error": "未找到该分析结果"}), 404
        return Response(
            json.dumps(r, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="run_{run_id}.json"'},
        )

    @app.get("/api/runs/<int:run_id>/export/pdf")
    def run_export_pdf(run_id: int):
        user = _auth_user(app, required=True)
        if user is None:
            return jsonify({"error": "未登录或登录已过期"}), 401
        if (not user.get("is_admin")) and (not user_owns_run(user["id"], run_id)):
            return jsonify({"error": "无权限导出该分析结果"}), 403
        r = get_analysis_run(run_id)
        if r is None:
            return jsonify({"error": "未找到该分析结果"}), 404
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.pdfgen import canvas
        except Exception:
            return jsonify({"error": "未安装 reportlab，请先执行：pip install reportlab"}), 400

        def _as_float(v):
            try:
                return float(v)
            except Exception:
                return None

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        page_w, page_h = A4
        y = page_h - 42

        font_name = "Helvetica"
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for fp in font_candidates:
            if os.path.exists(fp):
                try:
                    pdfmetrics.registerFont(TTFont("CN_FONT", fp))
                    font_name = "CN_FONT"
                    break
                except Exception:
                    pass

        def _new_page():
            nonlocal y
            c.showPage()
            y = page_h - 42

        def _set_font(size: int = 11):
            c.setFont(font_name, size)

        def _line(text: str, size: int = 11, gap: int = 16):
            nonlocal y
            if y < 64:
                _new_page()
            c.setFillColor(colors.HexColor("#0F172A"))
            _set_font(size)
            c.drawString(40, y, str(text)[:160])
            y -= gap

        def _section(title: str):
            nonlocal y
            if y < 90:
                _new_page()
            _line(title, size=12, gap=18)

        def _draw_trend_chart(history_rows: list, forecast_rows: list):
            nonlocal y
            series_hist = [(x.get("date"), _as_float(x.get("composite"))) for x in history_rows]
            series_hist = [x for x in series_hist if x[1] is not None]
            series_fc = [(x.get("date"), _as_float(x.get("forecast_composite"))) for x in forecast_rows]
            series_fc = [x for x in series_fc if x[1] is not None]
            if not series_hist and not series_fc:
                return

            # Zoom-in for readability: show only recent half of history.
            if len(series_hist) > 10:
                keep_hist = max(8, len(series_hist) // 2)
                series_hist = series_hist[-keep_hist:]

            if y < 260:
                _new_page()

            # Move chart slightly upward to make section 4/5 tighter.
            x0, y0, w, h = 40, y - 168, page_w - 80, 140
            all_vals = [v for _, v in series_hist] + [v for _, v in series_fc]
            vmin, vmax = min(all_vals), max(all_vals)
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1.0

            c.setStrokeColor(colors.HexColor("#CBD5E1"))
            c.rect(x0, y0, w, h, stroke=1, fill=0)
            _set_font(9)
            c.setFillColor(colors.HexColor("#334155"))
            c.drawString(x0, y0 + h + 8, "综合指数趋势图（历史实线，预测虚线）")
            c.drawRightString(x0 - 4, y0 + h - 2, f"{vmax:.3f}")
            c.drawRightString(x0 - 4, y0 - 2, f"{vmin:.3f}")

            all_points = series_hist + series_fc
            n = len(all_points)
            if n <= 1:
                y = y0 - 20
                return

            def _xy(idx: int, val: float):
                px = x0 + (idx / (n - 1)) * w
                py = y0 + ((val - vmin) / (vmax - vmin)) * h
                return px, py

            if len(series_hist) >= 2:
                c.setStrokeColor(colors.HexColor("#0D9488"))
                c.setLineWidth(1.6)
                p = c.beginPath()
                for i, (_, v) in enumerate(series_hist):
                    px, py = _xy(i, v)
                    if i == 0:
                        p.moveTo(px, py)
                    else:
                        p.lineTo(px, py)
                c.drawPath(p)

            if len(series_fc) >= 1:
                start = max(0, len(series_hist) - 1)
                c.setStrokeColor(colors.HexColor("#F59E0B"))
                c.setLineWidth(1.4)
                c.setDash(4, 2)
                p = c.beginPath()
                if len(series_hist) >= 1:
                    px, py = _xy(start, series_hist[-1][1])
                    p.moveTo(px, py)
                for i, (_, v) in enumerate(series_fc):
                    px, py = _xy(start + i + 1, v)
                    p.lineTo(px, py)
                c.drawPath(p)
                c.setDash()
            y = y0 - 16

        params = r.get("params", {}) or {}
        result_data = r.get("result", {}) or {}
        weights = result_data.get("weights", {}) or {}
        states = result_data.get("states_history", []) or []
        forecast_rows = result_data.get("future_states", []) or []
        hist_rows = result_data.get("composite_history", []) or []
        fc_raw = result_data.get("forecast", []) or []
        fusion_map = {
            "pca": "主成分分析（PCA）",
            "dfm": "动态因子模型（DFM）",
            "entropy": "熵权法",
            "equal": "等权法",
        }
        forecast_map = {"hw": "霍尔特-温特斯（HW）", "lstm": "长短期记忆网络（LSTM）"}

        # 智能结论特征
        items = sorted(weights.items(), key=lambda kv: (_as_float(kv[1]) if _as_float(kv[1]) is not None else -1), reverse=True)
        top3 = [x[0] for x in items[:3]]
        latest = states[-1] if states else {}
        current_state = latest.get("state_cn") or latest.get("state") or "未知"
        fc_vals = [_as_float(x.get("forecast_composite")) for x in fc_raw]
        fc_vals = [x for x in fc_vals if x is not None]
        if len(fc_vals) >= 2:
            slope = (fc_vals[-1] - fc_vals[0]) / (len(fc_vals) - 1)
        else:
            slope = 0.0
        if slope > 0.03:
            trend_text = "短期上行"
        elif slope < -0.03:
            trend_text = "短期下行"
        else:
            trend_text = "短期震荡"

        future_states = [x.get("state_cn") or x.get("state") or "-" for x in forecast_rows[:8]]
        shift_count = 0
        for i in range(1, len(future_states)):
            if future_states[i] != future_states[i - 1]:
                shift_count += 1
        risk_parts = []
        if shift_count >= 2:
            risk_parts.append("阶段切换较频繁")
        if abs(slope) < 0.01:
            risk_parts.append("趋势强度偏弱")
        if not risk_parts:
            risk_parts.append("短期风险可控")
        risk_text = "、".join(risk_parts)

        hist_vals = [_as_float(x.get("composite")) for x in hist_rows[-24:]]
        hist_vals = [x for x in hist_vals if x is not None]
        data_score = min(1.0, len(hist_vals) / 24.0)
        if len(hist_vals) >= 2:
            vol = float(np.std(hist_vals))
            stability_score = max(0.0, min(1.0, 1.0 - vol / 2.0))
        else:
            stability_score = 0.4
        consistency_score = max(0.0, min(1.0, 1.0 - shift_count / 4.0))
        confidence = int(round((0.4 * data_score + 0.3 * stability_score + 0.3 * consistency_score) * 100))
        if confidence >= 75:
            confidence_level = "高"
        elif confidence >= 55:
            confidence_level = "中"
        else:
            confidence_level = "低"

        # 标题横幅（恢复上一版）
        c.setFillColor(colors.HexColor("#0F766E"))
        c.roundRect(32, y - 16, page_w - 64, 34, 6, stroke=0, fill=1)
        c.setFillColor(colors.white)
        _set_font(15)
        c.drawString(42, y - 4, "宏观经济周期分析报告")
        y -= 36
        c.setFillColor(colors.HexColor("#0F172A"))

        # 概览信息卡片（恢复上一版）
        card_h = 56
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.roundRect(40, y - card_h + 8, page_w - 80, card_h, 6, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#334155"))
        _set_font(10)
        c.drawString(48, y - 8, f"报告编号：{run_id}")
        c.drawString(200, y - 8, f"数据集编号：{r.get('dataset_id')}")
        c.drawString(360, y - 8, f"创建时间：{(r.get('created_at') or '-')[:16]}")
        c.drawString(48, y - 26, f"融合方法：{fusion_map.get(params.get('fusion_method'), params.get('fusion_method', '-'))}")
        c.drawString(230, y - 26, f"预测模型：{forecast_map.get(params.get('forecast_model'), params.get('forecast_model', '-'))}")
        c.drawString(390, y - 26, f"预测期数：{params.get('horizon_months', '-')}")
        y -= 62

        _section("一、模型与参数")
        _line(f"标准化方式：{params.get('standardize', '-')}")
        _line(f"数据变换方式：{params.get('transform_type', '-')}")
        inv = params.get("inverse_columns", [])
        inv_text = ",".join(inv) if isinstance(inv, list) else (inv or "-")
        _line(f"逆向指标：{inv_text}")

        _section("二、指标权重结果")
        if not items:
            _line("无权重结果")
        else:
            if y < 160:
                _new_page()
            table_x = 44
            row_h = 18
            c.setFillColor(colors.HexColor("#E2E8F0"))
            c.rect(table_x, y - row_h + 4, page_w - 88, row_h, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#0F172A"))
            _set_font(10)
            c.drawString(table_x + 6, y - 9, "指标名称")
            c.drawString(table_x + 230, y - 9, "权重")
            c.drawString(table_x + 300, y - 9, "占比")
            y -= row_h
            for i, (k, v) in enumerate(items[:12]):
                if y < 74:
                    _new_page()
                if i % 2 == 0:
                    c.setFillColor(colors.HexColor("#F8FAFC"))
                    c.rect(table_x, y - row_h + 4, page_w - 88, row_h, stroke=0, fill=1)
                c.setFillColor(colors.HexColor("#0F172A"))
                fv = _as_float(v)
                c.drawString(table_x + 6, y - 9, str(k)[:40])
                c.drawRightString(table_x + 282, y - 9, "-" if fv is None else f"{fv:.4f}")
                c.drawRightString(table_x + 360, y - 9, "-" if fv is None else f"{fv * 100:.2f}%")
                y -= row_h

        y -= 14
        _section("三、周期状态与预测")
        _line(f"当前阶段：{current_state}")
        _line(f"当前日期：{latest.get('date') or '-'}")
        if forecast_rows:
            _line("未来预测（前 8 期）：", gap=14)
            if y < 170:
                _new_page()
            table_x = 44
            row_h = 18
            c.setFillColor(colors.HexColor("#E2E8F0"))
            c.rect(table_x, y - row_h + 4, page_w - 88, row_h, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#0F172A"))
            _set_font(10)
            c.drawString(table_x + 6, y - 9, "日期")
            c.drawString(table_x + 110, y - 9, "预测指数")
            c.drawString(table_x + 230, y - 9, "预测阶段")
            y -= row_h
            for i, row in enumerate(forecast_rows[:8]):
                if y < 74:
                    _new_page()
                if i % 2 == 0:
                    c.setFillColor(colors.HexColor("#F8FAFC"))
                    c.rect(table_x, y - row_h + 4, page_w - 88, row_h, stroke=0, fill=1)
                c.setFillColor(colors.HexColor("#0F172A"))
                v = _as_float(row.get("forecast_composite"))
                vtxt = "-" if v is None else f"{v:.4f}"
                c.drawString(table_x + 6, y - 9, str(row.get("date", "-"))[:16])
                c.drawString(table_x + 110, y - 9, vtxt)
                c.drawString(table_x + 230, y - 9, str(row.get("state_cn") or row.get("state") or "-")[:30])
                y -= row_h
        else:
            _line("未来阶段预测：无")

        y -= 8
        _section("四、趋势图")
        _draw_trend_chart(hist_rows, fc_raw)

        y -= 8
        _section("五、结论")
        top = items[0][0] if items else "-"
        c.setFillColor(colors.HexColor("#ECFEFF"))
        c.roundRect(40, y - 64, page_w - 80, 60, 5, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#0F172A"))
        _set_font(10)
        c.drawString(48, y - 20, f"主导指标：{top}；关键驱动：{'、'.join(top3) if top3 else '-'}；趋势判断：{trend_text}")
        c.drawString(48, y - 36, f"风险提示：{risk_text}；结论可信度：{confidence}/100（{confidence_level}）")
        c.drawString(48, y - 52, f"建议：当前处于{current_state}，建议结合阶段预测进行月度跟踪并滚动更新。")
        y -= 70

        # 页脚
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.line(40, 30, page_w - 40, 30)
        _set_font(9)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawString(40, 18, "Eco Cycle System · 自动生成分析报告")
        c.drawRightString(page_w - 40, 18, datetime.utcnow().strftime("%Y-%m-%d %H:%M"))

        c.save()
        buf.seek(0)
        return Response(
            buf.read(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="run_{run_id}.pdf"'},
        )

    # --- Admin APIs ---
    @app.get("/api/admin/users")
    def admin_users():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        return jsonify(list_users())

    @app.post("/api/admin/users")
    def admin_add_user():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", "")).strip()
        is_admin = bool(payload.get("is_admin", False))
        if (not _valid_username(username)) or len(password) < 6:
            return jsonify({"error": "用户名需为3-32位（中文/字母/数字/下划线），密码至少6位"}), 400
        try:
            uid = create_user(username, generate_password_hash(password), is_admin=is_admin)
        except ValueError as e:
            msg = str(e)
            if "username already exists" in msg:
                msg = "用户名已存在，请更换。"
            return jsonify({"error": msg}), 400
        add_audit_log("admin_add_user", user_id=user["id"], username=user["username"], detail=f"new_user={username}")
        return jsonify({"id": uid, "username": username, "is_admin": is_admin})

    @app.delete("/api/admin/users/<int:user_id>")
    def admin_del_user(user_id: int):
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        ok = delete_user(user_id)
        if not ok:
            return jsonify({"error": "用户不存在"}), 404
        add_audit_log("admin_delete_user", user_id=user["id"], username=user["username"], detail=f"user_id={user_id}")
        return jsonify({"ok": True})

    @app.get("/api/admin/stats/runs")
    def admin_stats_runs():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        return jsonify(get_user_run_stats())

    @app.get("/api/admin/configs")
    def admin_configs_list():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        return jsonify(list_system_configs())

    @app.post("/api/admin/configs")
    def admin_configs_set():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        payload = request.get_json(silent=True) or {}
        key = str(payload.get("key", "")).strip()
        if not key:
            return jsonify({"error": "配置键 key 不能为空"}), 400
        set_system_config(key, payload.get("value"))
        add_audit_log("admin_set_config", user_id=user["id"], username=user["username"], detail=f"key={key}")
        return jsonify({"ok": True})

    @app.get("/api/admin/models")
    def admin_models_list():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        model_type = request.args.get("model_type")
        return jsonify(list_models(model_type=model_type))

    @app.post("/api/admin/models")
    def admin_models_upsert():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        payload = request.get_json(silent=True) or {}
        model_type = str(payload.get("model_type", "")).strip()
        name = str(payload.get("name", "")).strip()
        if model_type not in {"fusion", "forecast"} or not name:
            return jsonify({"error": "model_type（fusion/forecast）和 name 不能为空"}), 400
        model_id = upsert_model(
            model_type=model_type,
            name=name,
            enabled=bool(payload.get("enabled", True)),
            params=payload.get("params", {}) or {},
        )
        add_audit_log("admin_upsert_model", user_id=user["id"], username=user["username"], detail=f"{model_type}:{name}")
        return jsonify({"id": model_id, "ok": True})

    @app.delete("/api/admin/models/<int:model_id>")
    def admin_models_delete(model_id: int):
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        ok = delete_model(model_id)
        if not ok:
            return jsonify({"error": "模型不存在"}), 404
        add_audit_log("admin_delete_model", user_id=user["id"], username=user["username"], detail=f"model_id={model_id}")
        return jsonify({"ok": True})

    @app.get("/api/admin/logs")
    def admin_logs():
        user = _auth_user(app, required=True)
        if not _ensure_admin(user):
            return jsonify({"error": "仅管理员可访问"}), 403
        lines = int(request.args.get("lines", "200"))
        lines = max(20, min(lines, 2000))
        logs = list_audit_logs(limit=lines)
        return jsonify({"logs": logs})

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
