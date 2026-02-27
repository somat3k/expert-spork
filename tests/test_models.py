"""Tests for AI/ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.models.anomaly_detection import AnomalyDetector
from eth_algo_trading.models.hyperparameter_tuner import AdaptiveConfig, PerformanceTracker
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


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class TestPerformanceTracker:
    def test_initial_state(self):
        tracker = PerformanceTracker()
        assert tracker.n_recorded == 0
        assert tracker.rolling_accuracy() == 0.0
        assert tracker.rolling_confidence() == 0.0

    def test_all_correct_accuracy(self):
        tracker = PerformanceTracker()
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.8)
        assert tracker.rolling_accuracy() == 1.0

    def test_all_wrong_accuracy(self):
        tracker = PerformanceTracker()
        for _ in range(10):
            tracker.record(predicted=0, actual=1, confidence=0.6)
        assert tracker.rolling_accuracy() == 0.0

    def test_mixed_accuracy(self):
        tracker = PerformanceTracker()
        for _ in range(5):
            tracker.record(predicted=1, actual=1, confidence=0.7)
        for _ in range(5):
            tracker.record(predicted=0, actual=1, confidence=0.4)
        assert abs(tracker.rolling_accuracy() - 0.5) < 1e-9

    def test_rolling_confidence(self):
        tracker = PerformanceTracker()
        tracker.record(predicted=1, actual=1, confidence=0.6)
        tracker.record(predicted=1, actual=1, confidence=0.8)
        assert abs(tracker.rolling_confidence() - 0.7) < 1e-9

    def test_window_respected(self):
        cfg = AdaptiveConfig(window=5)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=1, actual=0, confidence=0.5)  # wrong
        # Only 5 most-recent entries kept
        assert tracker.n_recorded == 5
        assert tracker.rolling_accuracy() == 0.0

    def test_suggest_n_estimators_increases_on_low_accuracy(self):
        cfg = AdaptiveConfig(window=10, accuracy_target=0.6, estimators_step=25)
        tracker = PerformanceTracker(cfg)
        # Fill window with wrong predictions so accuracy = 0.0
        for _ in range(10):
            tracker.record(predicted=0, actual=1, confidence=0.5)
        new_val = tracker.suggest_n_estimators(200)
        assert new_val == 225

    def test_suggest_n_estimators_decreases_on_high_accuracy_and_confidence(self):
        cfg = AdaptiveConfig(window=10, accuracy_target=0.6, confidence_target=0.5, estimators_step=25)
        tracker = PerformanceTracker(cfg)
        # All correct with high confidence → accuracy=1.0, confidence=0.9
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.9)
        new_val = tracker.suggest_n_estimators(200)
        assert new_val == 175

    def test_suggest_n_estimators_no_change_below_half_window(self):
        cfg = AdaptiveConfig(window=10)
        tracker = PerformanceTracker(cfg)
        # Only 4 out of 10 required (window//2)
        for _ in range(4):
            tracker.record(predicted=0, actual=1, confidence=0.5)
        assert tracker.suggest_n_estimators(200) == 200

    def test_suggest_n_estimators_clamps_to_max(self):
        cfg = AdaptiveConfig(window=10, accuracy_target=0.6, estimators_step=50, estimators_max=250)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=0, actual=1, confidence=0.5)
        assert tracker.suggest_n_estimators(250) == 250

    def test_suggest_n_estimators_clamps_to_min(self):
        cfg = AdaptiveConfig(window=10, accuracy_target=0.5, confidence_target=0.5, estimators_step=50, estimators_min=100)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.9)
        assert tracker.suggest_n_estimators(100) == 100

    def test_suggest_contamination_increases_when_rate_higher(self):
        cfg = AdaptiveConfig(window=10, contamination_step=0.01)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.8)
        # observed rate 1.0 (all anomalous) > current 0.05 → increase
        new_val = tracker.suggest_contamination(current=0.05, observed_anomaly_rate=0.20)
        assert abs(new_val - 0.06) < 1e-9

    def test_suggest_contamination_decreases_when_rate_lower(self):
        cfg = AdaptiveConfig(window=10, contamination_step=0.01)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.8)
        # observed rate 0.01 < current 0.10 → decrease
        new_val = tracker.suggest_contamination(current=0.10, observed_anomaly_rate=0.01)
        assert abs(new_val - 0.09) < 1e-9

    def test_suggest_contamination_no_change_within_step(self):
        cfg = AdaptiveConfig(window=10, contamination_step=0.02)
        tracker = PerformanceTracker(cfg)
        for _ in range(10):
            tracker.record(predicted=1, actual=1, confidence=0.8)
        # Delta < step → no change
        new_val = tracker.suggest_contamination(current=0.10, observed_anomaly_rate=0.11)
        assert abs(new_val - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# AnomalyDetector adaptive behaviour
# ---------------------------------------------------------------------------

class TestAnomalyDetectorAdaptive:
    def _make_features(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "net_inflow": rng.normal(0, 100, n),
            "gas_price": rng.uniform(10, 100, n),
            "whale_transfers": rng.poisson(5, n).astype(float),
        })

    def test_rolling_accuracy_and_confidence_start_zero(self):
        detector = AnomalyDetector()
        assert detector.rolling_accuracy == 0.0
        assert detector.rolling_confidence == 0.0

    def test_record_outcome_updates_rolling_accuracy(self):
        cfg = AdaptiveConfig(window=5)
        detector = AnomalyDetector(adaptive_config=cfg)
        features = self._make_features()
        detector.fit(features)
        # Make a prediction, then record an outcome
        was_anomaly = detector.predict(features)
        detector.record_outcome(was_anomaly)  # correct outcome
        assert detector.rolling_accuracy == 1.0

    def test_record_outcome_before_predict_is_noop(self):
        detector = AnomalyDetector()
        detector.fit(self._make_features())
        # No predict called yet — record_outcome should not raise
        detector.record_outcome(True)
        assert detector.rolling_accuracy == 0.0

    def test_contamination_adjusts_toward_observed_rate(self):
        cfg = AdaptiveConfig(window=10, contamination_step=0.01)
        detector = AnomalyDetector(contamination=0.05, adaptive_config=cfg)
        features = self._make_features()
        detector.fit(features)
        initial_contamination = detector.contamination
        # Simulate all outcomes being anomalous → observed_rate = 1.0 >> 0.05
        for _ in range(10):
            detector.predict(features)
            detector.record_outcome(True)
        expected = initial_contamination + cfg.contamination_step
        assert abs(detector.contamination - expected) < 1e-9

    def test_adapt_no_change_without_history(self):
        detector = AnomalyDetector(contamination=0.05)
        before = detector.contamination
        detector.adapt()
        assert detector.contamination == before
