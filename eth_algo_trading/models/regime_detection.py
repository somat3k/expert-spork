"""
Market regime detection model.

Uses a Gaussian Hidden Markov Model (via scikit-learn) or a lightweight
rule-based fallback to classify the current market regime as
'bull', 'sideways', or 'bear'.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


_REGIME_LABELS = {0: "bull", 1: "sideways", 2: "bear"}


class RegimeDetector:
    """
    Hidden Markov Model-based regime classifier.

    Parameters
    ----------
    n_states:
        Number of hidden states (default 3: bull / sideways / bear).
    """

    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states
        self._model = None
        self._is_fitted = False

    def _build_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
        returns = ohlcv["close"].pct_change().dropna()
        vol = returns.rolling(5).std().dropna()
        aligned = returns.align(vol, join="inner")[0]
        aligned_vol = returns.align(vol, join="inner")[1]
        return np.column_stack([aligned.values, aligned_vol.values])

    def fit(self, ohlcv: pd.DataFrame) -> "RegimeDetector":
        """Fit the HMM on historical OHLCV data."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("hmmlearn is required for RegimeDetector.fit()") from exc

        X = self._build_features(ohlcv)
        self._model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100)
        self._model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, ohlcv: pd.DataFrame) -> str:
        """
        Predict the current regime label.

        Falls back to a rule-based classifier when the model is not fitted.
        """
        if not self._is_fitted or self._model is None:
            return self._rule_based(ohlcv)

        X = self._build_features(ohlcv)
        if len(X) == 0:
            return "sideways"

        state = int(self._model.predict(X)[-1])
        return _REGIME_LABELS.get(state % self.n_states, "sideways")

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based(ohlcv: pd.DataFrame) -> str:
        """Simple 50-bar trend rule used when the HMM is not fitted."""
        if len(ohlcv) < 50:
            return "sideways"
        ret_50 = ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-50] - 1
        if ret_50 > 0.10:
            return "bull"
        if ret_50 < -0.10:
            return "bear"
        return "sideways"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
