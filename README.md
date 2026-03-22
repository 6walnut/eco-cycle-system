# Eco Cycle System (Prototype)

This project implements an **economic cycle state analysis and visualization system** based on **fused China macroeconomic indicators**:

1. **Macro data acquisition & preprocessing**
   - CSV ingestion (`date` + indicator columns)
   - missing value interpolation
   - outlier clipping
   - optional transforms: `mom` / `yoy`
2. **Economic cycle multi-indicator fusion**
   - standardization: `zscore` / `minmax`
   - fusion: `equal` / `pca` / `entropy` weights
   - smoothing for a stable composite cycle index
3. **Visualization**
   - composite index line
   - phase state coloring (Expansion / Peak / Contraction / Trough)
   - indicator contribution proxy (last period)
   - phase timeline segments
4. **Forecasting**
   - 3–6 months forecast using Holt-Winters (with fallback)
   - future phase classification from forecasted composite values

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Flask REST API

Run:

```bash
python api_server.py
```

Endpoints:
- `GET /api/health` : health check
- `GET /api/sample` : return sample data preview
- `POST /api/analyze` : upload CSV and get analysis results (composite index, phase states, forecast)

`POST /api/analyze` expects `multipart/form-data`:
- `file` : CSV file (required unless `use_sample=true`)
- `use_sample` : set to `true` to analyze generated sample data
- `inverse_columns` : comma-separated indicator column names that should be sign-inverted (e.g. `unemployment`)

Optional form fields (defaults in parentheses):
- `transform_type` (`none`) : `none` / `mom` / `yoy`
- `standardize` (`zscore`) : `zscore` / `minmax`
- `fusion_method` (`pca`) : `equal` / `pca` / `entropy`
- `smoothing_window` (`3`)
- `horizon_months` (`3`)
- `ma_window` (`6`)
- `band_multiplier` (`0.3`)
- `clip_lo` (`0.01`), `clip_hi` (`0.99`)
- `start_date`, `end_date` : e.g. `2018-01-01`

Response: JSON containing `weights`, `composite_history`, `states_history`, `forecast`, `future_states`.

## CSV format

Upload a CSV with:
- `date`: monthly date (e.g., `2018-01-01`)
- other columns: macro indicators (any names)

Example header:

```csv
date,gdp,industrial_production,cpi,m2,credit,unemployment
```

