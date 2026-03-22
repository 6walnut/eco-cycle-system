import io
from typing import List, Optional

import pandas as pd
from flask import Flask, jsonify, request

from core import FusionConfig, create_sample_data, load_user_csv, run_analysis


def _parse_inverse_columns(raw: Optional[str], all_indicators: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return []
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    # Keep only indicators that exist.
    return [c for c in cols if c in all_indicators]


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/sample")
    def sample():
        df = create_sample_data()
        return jsonify({"csv_preview": df.head(10).to_dict(orient="records"), "rows": len(df)})

    @app.post("/api/analyze")
    def analyze():
        # Multipart: form-data with `file` (CSV).
        # Or use_sample=true to analyze generated sample data.
        use_sample = str(request.form.get("use_sample", "false")).lower() in {"1", "true", "yes"}
        uploaded = request.files.get("file")

        transform_type = request.form.get("transform_type", "none")
        standardize = request.form.get("standardize", "zscore")
        fusion_method = request.form.get("fusion_method", "pca")
        smoothing_window = int(request.form.get("smoothing_window", "3"))
        horizon_months = int(request.form.get("horizon_months", "3"))
        ma_window = int(request.form.get("ma_window", "6"))
        band_multiplier = float(request.form.get("band_multiplier", "0.3"))

        clip_lo = float(request.form.get("clip_lo", "0.01"))
        clip_hi = float(request.form.get("clip_hi", "0.99"))

        start_date = request.form.get("start_date")  # e.g. "2018-01-01"
        end_date = request.form.get("end_date")

        df: pd.DataFrame
        if use_sample:
            df = create_sample_data()
        else:
            if uploaded is None:
                return jsonify({"error": "Missing CSV file: provide form-data field `file` or set use_sample=true."}), 400
            df = load_user_csv(uploaded)

        indicator_cols = [c for c in df.columns if c != "date"]
        inverse_columns = _parse_inverse_columns(request.form.get("inverse_columns"), indicator_cols)

        config = FusionConfig(
            transform_type=transform_type,
            standardize=standardize,
            fusion_method=fusion_method,
            inverse_columns=inverse_columns,
            smoothing_window=smoothing_window,
        )

        try:
            result = run_analysis(
                df,
                config=config,
                clip_quantiles=(clip_lo, clip_hi),
                horizon_months=horizon_months,
                ma_window=ma_window,
                band_multiplier=band_multiplier,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(result)

    return app


if __name__ == "__main__":
    flask_app = create_app()
    # Default port: 5000
    flask_app.run(host="0.0.0.0", port=5000, debug=True)

