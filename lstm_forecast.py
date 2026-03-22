"""
LSTM-based multi-step forecast for the composite index (univariate).
Requires: pip install tensorflow (see requirements-ml.txt)
Falls back to Holt-Winters if tensorflow missing or data too short.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _try_import_tf():
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def forecast_composite_lstm(
    composite: pd.Series,
    horizon_months: int,
    lookback: int = 12,
    epochs: int = 40,
    units: int = 32,
) -> Tuple[pd.Series, Optional[pd.Series], dict]:
    """
    Train a small LSTM on scaled composite, recursive multi-step forecast.

    Returns:
      forecast_series, None, meta dict with keys: model, note
    """
    from sklearn.preprocessing import MinMaxScaler

    y = composite.dropna().astype(float)
    meta = {"model": "lstm", "note": ""}

    min_train = lookback + 24
    if len(y) < min_train or not _try_import_tf():
        meta["model"] = "hw_fallback"
        meta["note"] = (
            "TensorFlow not installed or series too short; use Holt-Winters instead."
            if not _try_import_tf()
            else f"Need at least {min_train} points for LSTM; got {len(y)}."
        )
        # Late import avoids circular import with core
        from core import forecast_composite

        return forecast_composite(composite, horizon_months)[0], None, meta

    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    arr = y.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    s = scaler.fit_transform(arr).flatten()

    X_list, Y_list = [], []
    for i in range(len(s) - lookback):
        X_list.append(s[i : i + lookback])
        Y_list.append(s[i + lookback])
    X_arr = np.array(X_list, dtype=np.float32).reshape(-1, lookback, 1)
    Y_arr = np.array(Y_list, dtype=np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(units, dropout=0.1),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    model.fit(X_arr, Y_arr, epochs=epochs, batch_size=min(16, len(X_arr)), verbose=0)

    # Recursive forecast
    window = s[-lookback:].copy()
    preds_scaled = []
    for _ in range(horizon_months):
        x_in = window.reshape(1, lookback, 1)
        p = float(model.predict(x_in, verbose=0)[0, 0])
        preds_scaled.append(p)
        window = np.append(window[1:], p)

    inv = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    last_date = y.index.max()
    idx = pd.date_range(last_date, periods=horizon_months + 1, freq="MS")[1:]
    series = pd.Series(inv, index=idx, name="forecast_lstm")
    meta["note"] = f"trained on {len(y)} points, lookback={lookback}"
    return series, None, meta
