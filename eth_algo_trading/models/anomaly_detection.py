"""
Anomaly detection model.

Uses an Isolation Forest to flag unusual exchange inflows, whale
transactions, or gas spikes that may precede large price moves.

Supports dynamic hyperparameter adaptation: call ``record_outcome()``
after each prediction to let the model track its observed anomaly rate
and automatically nudge the ``contamination`` parameter toward the
empirical rate via :meth:`adapt`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from eth_algo_trading.models.hyperparameter_tuner import AdaptiveConfig, PerformanceTracker


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for on-chain and market-microstructure data.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies in training data.
    n_estimators:
        Number of trees in the isolation forest.
    adaptive_config:
        Configuration for the dynamic hyperparameter tuner.  Pass
        ``None`` (default) to use default :class:`AdaptiveConfig` values.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        adaptive_config: Optional[AdaptiveConfig] = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self._is_fitted = False
        self._tracker = PerformanceTracker(adaptive_config)
        # Note: these two attributes are NOT thread-safe. Callers must ensure
        # that predict() and record_outcome() are invoked sequentially.
        self._last_prediction: Optional[int] = None
        self._last_anomaly_score: float = 0.0

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """
        Train the Isolation Forest on historical feature data.

        Recreates the underlying model using the current values of
        ``contamination`` and ``n_estimators``, so any changes made by
        :meth:`adapt` are picked up automatically on the next call to
        this method.
        """
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
        )
        self._model.fit(features.fillna(0))
        self._is_fitted = True
        return self

    def score(self, features: pd.DataFrame) -> float:
        """
        Return an anomaly score in [0, 1] for the latest row.

        Higher values indicate a more anomalous observation.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        row = features.fillna(0).iloc[[-1]]
        # IsolationForest.decision_function returns negative scores for anomalies
        raw = float(self._model.decision_function(row)[0])
        # Map to [0, 1]: raw is typically in [-0.5, 0.5]
        return float(np.clip(0.5 - raw, 0.0, 1.0))

    def predict(self, features: pd.DataFrame) -> bool:
        """Return True if the latest row is classified as anomalous."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        row = features.fillna(0).iloc[[-1]]
        is_anomaly = bool(self._model.predict(row)[0] == -1)
        self._last_prediction = 1 if is_anomaly else 0
        # Store anomaly score as confidence proxy: higher score → more anomalous
        self._last_anomaly_score = float(np.clip(
            0.5 - self._model.decision_function(row)[0], 0.0, 1.0
        ))
        return is_anomaly

    # ------------------------------------------------------------------
    # Dynamic hyperparameter adaptation
    # ------------------------------------------------------------------

    def record_outcome(self, was_anomaly: bool) -> None:
        """
        Record the ground-truth label for the most recent prediction.

        Parameters
        ----------
        was_anomaly:
            ``True`` if the observation turned out to be a genuine anomaly,
            ``False`` otherwise.

        After recording, :meth:`adapt` is automatically called to keep
        ``contamination`` aligned with the empirical anomaly rate.

        Raises
        ------
        RuntimeError
            If called before any prediction has been made (i.e. :meth:`predict`
            has not been called yet so there is no stored prediction to attach
            this outcome to).
        """
        if self._last_prediction is None:
            raise RuntimeError(
                "record_outcome() called before any prediction was made. "
                "Call predict() first so there is a stored prediction to "
                "attach this outcome to."
            )
        actual = 1 if was_anomaly else 0
        # Use the stored anomaly score as a proxy for how confident the model
        # was in its prediction: high score → confident it's anomalous;
        # low score → confident it's normal.
        confidence = (
            self._last_anomaly_score if self._last_prediction == 1
            else 1.0 - self._last_anomaly_score
        )
        self._tracker.record(
            predicted=self._last_prediction,
            actual=actual,
            confidence=confidence,
        )
        self.adapt()

    def adapt(self) -> None:
        """
        Adjust ``contamination`` based on the observed anomaly rate.

        The observed anomaly rate is the fraction of positive (anomalous)
        outcomes recorded in the tracker's window.  The new value is stored
        on the detector instance.  Call :meth:`fit` afterwards to recreate
        the underlying Isolation Forest with the updated parameter.
        """
        actuals = self._tracker.get_actuals()
        if not actuals:
            return
        observed_rate = sum(actuals) / len(actuals)
        self.contamination = self._tracker.suggest_contamination(
            self.contamination, observed_rate
        )

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
