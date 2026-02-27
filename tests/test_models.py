"""Tests for AI/ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.models.anomaly_detection import AnomalyDetector
from eth_algo_trading.models.regime_detection import RegimeDetector


def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0, 20, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 10, "low": close - 10, "close": close, "volume": np.ones(n) * 1000},
        index=idx,
    )


class TestAnomalyDetector:
    def _make_features(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "net_inflow": rng.normal(0, 100, n),
            "gas_price": rng.uniform(10, 100, n),
            "whale_transfers": rng.poisson(5, n).astype(float),
        })

    def test_fit_and_score(self):
        detector = AnomalyDetector(contamination=0.05)
        features = self._make_features(100)
        detector.fit(features)
        score = detector.score(features)
        assert 0.0 <= score <= 1.0

    def test_fit_and_predict(self):
        detector = AnomalyDetector(contamination=0.05)
        features = self._make_features(100)
        detector.fit(features)
        result = detector.predict(features)
        assert isinstance(result, bool)

    def test_not_fitted_raises(self):
        detector = AnomalyDetector()
        features = self._make_features(10)
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.score(features)

    def test_is_fitted_property(self):
        detector = AnomalyDetector()
        assert not detector.is_fitted
        detector.fit(self._make_features())
        assert detector.is_fitted

    def test_extreme_anomaly_gets_high_score(self):
        detector = AnomalyDetector(contamination=0.05)
        normal = self._make_features(200)
        detector.fit(normal)
        # Craft an obviously anomalous row
        anomaly = pd.DataFrame({
            "net_inflow": [1_000_000.0],
            "gas_price": [1_000_000.0],
            "whale_transfers": [1_000.0],
        })
        score = detector.score(anomaly)
        # High anomaly score expected (> 0.5)
        assert score > 0.5


class TestRegimeDetector:
    def test_rule_based_bull(self):
        n = 60
        close = np.linspace(1000, 1150, n)  # +15 % → bull
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame({"close": close}, index=idx)
        regime = RegimeDetector._rule_based(ohlcv)
        assert regime == "bull"

    def test_rule_based_bear(self):
        n = 60
        close = np.linspace(2000, 1700, n)  # -15 % → bear
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame({"close": close}, index=idx)
        regime = RegimeDetector._rule_based(ohlcv)
        assert regime == "bear"

    def test_rule_based_sideways(self):
        n = 60
        close = np.linspace(2000, 2050, n)  # +2.5 % → sideways
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame({"close": close}, index=idx)
        regime = RegimeDetector._rule_based(ohlcv)
        assert regime == "sideways"

    def test_rule_based_insufficient_data(self):
        ohlcv = _make_ohlcv(10)
        regime = RegimeDetector._rule_based(ohlcv)
        assert regime == "sideways"

    def test_predict_without_fit(self):
        detector = RegimeDetector()
        ohlcv = _make_ohlcv()
        regime = detector.predict(ohlcv)
        assert regime in {"bull", "sideways", "bear"}

    def test_is_fitted_starts_false(self):
        assert not RegimeDetector().is_fitted
