import io
from typing import List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from core import FusionConfig, _parse_date_col, create_sample_data, load_user_csv, run_analysis
from db_models import get_analysis_run, init_db, list_datasets, load_dataset_rows, save_analysis_run, save_dataset


def _parse_inverse_columns(raw: Optional[str], all_indicators: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return []
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return [c for c in cols if c in all_indicators]


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


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_db()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/sample")
    def sample():
        df = create_sample_data()
        return jsonify({"csv_preview": df.head(10).to_dict(orient="records"), "rows": len(df)})

    @app.post("/api/analyze")
    def analyze():
        use_sample = str(request.form.get("use_sample", "false")).lower() in {"1", "true", "yes"}
        uploaded = request.files.get("file")
        params = _analyze_params_from_form()

        if use_sample:
            df = create_sample_data()
        else:
            if uploaded is None:
                return jsonify({"error": "Missing CSV file: provide form-data field `file` or set use_sample=true."}), 400
            df = load_user_csv(uploaded)

        indicator_cols = [c for c in df.columns if c != "date"]
        inverse_columns = _parse_inverse_columns(request.form.get("inverse_columns"), indicator_cols)

        try:
            result = _run_analyze_df(df, params, inverse_columns)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(result)

    @app.post("/api/datasets")
    def upload_dataset():
        """Upload CSV and store rows in DB. Returns dataset id."""
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "Missing file"}), 400
        name = request.form.get("name", "upload")
        raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
        df = _parse_date_col(df)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.replace({np.nan: None, pd.NA: None})
        rows = df.to_dict(orient="records")
        ds_id = save_dataset(rows, name=name)
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
        result["dataset_id"] = ds_id
        result["run_id"] = run_id
        return jsonify(result)

    @app.get("/api/runs/<int:run_id>")
    def run_get(run_id: int):
        r = get_analysis_run(run_id)
        if r is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(r)

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
