"""
Anomaly detection model.

Uses an Isolation Forest to flag unusual exchange inflows, whale
transactions, or gas spikes that may precede large price moves.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Isolation Forest anomaly detector for on-chain and market-microstructure data.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies in training data.
    n_estimators:
        Number of trees in the isolation forest.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100) -> None:
        self.contamination = contamination
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """Train the Isolation Forest on historical feature data."""
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
        return bool(self._model.predict(row)[0] == -1)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
