"""
Price forecasting model.

Uses gradient-boosted trees (XGBoost) trained on engineered features derived
from OHLCV data and optional supplementary signals (funding rates, on-chain
metrics, sentiment).  The model predicts the *direction* of the next
``forecast_horizon`` bars (up / down / flat).

Supports dynamic hyperparameter adaptation: after each prediction is resolved,
call ``record_outcome()`` with the actual direction and the model will
automatically adjust ``n_estimators`` (and optionally retrain) based on its
rolling accuracy and confidence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from eth_algo_trading.models.hyperparameter_tuner import AdaptiveConfig, PerformanceTracker


def _make_features(ohlcv: pd.DataFrame, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build a feature matrix from OHLCV bars."""
    df = pd.DataFrame(index=ohlcv.index)

    close = ohlcv["close"]
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_12"] = close.pct_change(12)
    df["log_volume"] = np.log1p(ohlcv["volume"])
    df["hl_range"] = (ohlcv["high"] - ohlcv["low"]) / close
    df["close_vs_ema20"] = close / close.ewm(span=20).mean() - 1

    if extra is not None:
        for col in extra.columns:
            if col not in df.columns:
                df[col] = extra[col]

    return df.dropna()


class PriceForecaster:
    """
    XGBoost-based price-direction forecaster.

    Parameters
    ----------
    forecast_horizon:
        Number of bars ahead to forecast.
    n_estimators:
        Number of gradient-boosted trees.
    adaptive_config:
        Configuration for the dynamic hyperparameter tuner.  Pass
        ``None`` (default) to use default :class:`AdaptiveConfig` values.
    """

    def __init__(
        self,
        forecast_horizon: int = 12,
        n_estimators: int = 200,
        adaptive_config: Optional[AdaptiveConfig] = None,
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.n_estimators = n_estimators
        self._model = None
        self._is_fitted = False
        self._tracker = PerformanceTracker(adaptive_config)
        # Store the last prediction (class index 0/1/2) and its confidence
        # so that record_outcome() can pair them with the actual result.
        self._last_prediction: Optional[int] = None
        self._last_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ohlcv: pd.DataFrame,
        extra: Optional[pd.DataFrame] = None,
    ) -> "PriceForecaster":
        """Train the forecaster on historical OHLCV data."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for PriceForecaster.fit()") from exc

        X = _make_features(ohlcv, extra)
        future_return = ohlcv["close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        y = np.sign(future_return.reindex(X.index)).fillna(0).astype(int) + 1  # 0/1/2

        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        ohlcv: pd.DataFrame,
        extra: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float, float]:
        """
        Return ``(prob_down, prob_flat, prob_up)`` for the latest bar.

        Returns
        -------
        Tuple[float, float, float]
            Probabilities for direction classes –1, 0, +1.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X = _make_features(ohlcv, extra)
        if X.empty:
            return (1 / 3, 1 / 3, 1 / 3)

        proba = self._model.predict_proba(X.iloc[[-1]])[0]
        self._last_prediction = int(np.argmax(proba))
        self._last_confidence = float(proba[self._last_prediction])
        return tuple(float(p) for p in proba)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dynamic hyperparameter adaptation
    # ------------------------------------------------------------------

    def record_outcome(self, actual_direction: int) -> None:
        """
        Record the ground-truth outcome for the most recent prediction.

        Parameters
        ----------
        actual_direction:
            Actual direction class: 0 (down), 1 (flat), or 2 (up).
            This is the same encoding used internally by the model and
            returned as the argmax of ``predict_proba``.

        After recording, :meth:`adapt` is automatically called so that
        ``n_estimators`` stays calibrated to recent performance.
        """
        if self._last_prediction is None:
            return
        self._tracker.record(
            predicted=self._last_prediction,
            actual=actual_direction,
            confidence=self._last_confidence,
        )
        self.adapt()

    def adapt(self) -> None:
        """
        Adjust ``n_estimators`` based on rolling accuracy and confidence.

        The new value is stored on the instance.  On the next call to
        :meth:`fit` the updated ``n_estimators`` will be used automatically.
        """
        self.n_estimators = self._tracker.suggest_n_estimators(self.n_estimators)

    @property
    def rolling_accuracy(self) -> float:
        """Rolling prediction accuracy over the tracker window."""
        return self._tracker.rolling_accuracy()

    @property
    def rolling_confidence(self) -> float:
        """Rolling mean confidence over the tracker window."""
        return self._tracker.rolling_confidence()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
