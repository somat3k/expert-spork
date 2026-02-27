"""
Tests for the Hyperliquid execution channel.

All tests run without a live Hyperliquid connection; the SDK client is
replaced with lightweight mocks so that decision-logic and paper-trading
paths can be validated in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.execution.hyperliquid_executor import (
    HyperliquidExecutor,
    HyperliquidOrderResult,
)
from eth_algo_trading.risk.manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 2000.0 * np.cumprod(1 + trend + rng.normal(0, 0.005, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n) * 1000},
        index=idx,
    )


def _paper_executor(**kwargs) -> HyperliquidExecutor:
    """Create a paper-trading executor with SDK disabled."""
    with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
        return HyperliquidExecutor(paper_trading=True, **kwargs)


# ---------------------------------------------------------------------------
# HyperliquidOrderResult dataclass
# ---------------------------------------------------------------------------

class TestHyperliquidOrderResult:
    def test_fields(self):
        r = HyperliquidOrderResult(
            coin="ETH", side="buy", size=0.1, price=2000.0,
            status="paper", signal_direction="long", signal_strength=0.8,
        )
        assert r.coin == "ETH"
        assert r.raw_response == {}

    def test_custom_raw_response(self):
        r = HyperliquidOrderResult(
            coin="BTC", side="sell", size=0.01, price=50000.0,
            status="filled", signal_direction="short", signal_strength=0.6,
            raw_response={"status": "ok"},
        )
        assert r.raw_response == {"status": "ok"}


# ---------------------------------------------------------------------------
# HyperliquidExecutor initialisation
# ---------------------------------------------------------------------------

class TestHyperliquidExecutorInit:
    def test_no_sdk_graceful_degradation(self):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor()
        assert ex._exchange is None
        assert ex._info is None

    def test_paper_trading_flag(self):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor(paper_trading=True)
        assert ex.paper_trading is True


# ---------------------------------------------------------------------------
# Market data helpers (offline)
# ---------------------------------------------------------------------------

class TestMarketDataOffline:
    def setup_method(self):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            self.ex = HyperliquidExecutor()

    def test_fetch_mid_price_returns_zero_without_sdk(self):
        assert self.ex.fetch_mid_price("ETH") == 0.0

    def test_fetch_account_value_returns_zero_without_sdk(self):
        assert self.ex.fetch_account_value() == 0.0

    def test_get_position_size_returns_zero_without_sdk(self):
        assert self.ex.get_position_size("ETH") == 0.0


class TestMarketDataWithMockedSDK:
    def _executor_with_mock_info(self, mids=None, user_state=None):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor(account_address="0xABCD")
        mock_info = MagicMock()
        mock_info.all_mids.return_value = mids or {"ETH": "2500.0"}
        mock_info.user_state.return_value = user_state or {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "1.5"}}],
        }
        ex._info = mock_info
        return ex

    def test_fetch_mid_price(self):
        ex = self._executor_with_mock_info()
        assert ex.fetch_mid_price("ETH") == pytest.approx(2500.0)

    def test_fetch_mid_price_missing_coin(self):
        ex = self._executor_with_mock_info(mids={"BTC": "50000.0"})
        assert ex.fetch_mid_price("ETH") == 0.0

    def test_fetch_account_value(self):
        ex = self._executor_with_mock_info()
        assert ex.fetch_account_value() == pytest.approx(10000.0)

    def test_get_position_size_long(self):
        ex = self._executor_with_mock_info()
        assert ex.get_position_size("ETH") == pytest.approx(1.5)

    def test_get_position_size_not_found(self):
        ex = self._executor_with_mock_info()
        assert ex.get_position_size("BTC") == 0.0

    def test_fetch_mid_price_exception_returns_zero(self):
        ex = self._executor_with_mock_info()
        ex._info.all_mids.side_effect = RuntimeError("network error")
        assert ex.fetch_mid_price("ETH") == 0.0


# ---------------------------------------------------------------------------
# compute_signal
# ---------------------------------------------------------------------------

class TestComputeSignal:
    def setup_method(self):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            self.ex = HyperliquidExecutor(min_prob_threshold=0.45, max_anomaly_score=0.70)

    def test_flat_when_insufficient_data(self):
        ohlcv = _make_ohlcv(5)
        direction, strength = self.ex.compute_signal(ohlcv)
        assert direction == "flat"

    def test_long_on_uptrend(self):
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        direction, strength = self.ex.compute_signal(ohlcv)
        assert direction == "long"
        assert 0 < strength <= 1.0

    def test_short_on_downtrend(self):
        close = np.linspace(1200, 1000, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        direction, strength = self.ex.compute_signal(ohlcv)
        assert direction == "short"

    def test_fitted_forecaster_used(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.10, 0.10, 0.80)
        ohlcv = _make_ohlcv()
        direction, strength = self.ex.compute_signal(ohlcv, forecaster=mock_forecaster)
        assert direction == "long"
        assert strength == pytest.approx(0.80)

    def test_unfitted_forecaster_falls_through_to_naive(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = False
        # Strong uptrend for naive fallback
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        direction, _ = self.ex.compute_signal(ohlcv, forecaster=mock_forecaster)
        assert direction == "long"

    def test_forecaster_flat_when_below_threshold(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.35, 0.35, 0.30)
        direction, _ = self.ex.compute_signal(_make_ohlcv(), forecaster=mock_forecaster)
        assert direction == "flat"

    def test_bear_regime_suppresses_long(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.10, 0.10, 0.80)

        mock_regime = MagicMock()
        mock_regime.predict.return_value = "bear"

        direction, _ = self.ex.compute_signal(
            _make_ohlcv(), forecaster=mock_forecaster, regime_detector=mock_regime
        )
        assert direction == "flat"

    def test_bull_regime_suppresses_short(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.80, 0.10, 0.10)

        mock_regime = MagicMock()
        mock_regime.predict.return_value = "bull"

        direction, _ = self.ex.compute_signal(
            _make_ohlcv(), forecaster=mock_forecaster, regime_detector=mock_regime
        )
        assert direction == "flat"

    def test_anomaly_gate_suppresses_signal(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.10, 0.10, 0.80)

        mock_anomaly = MagicMock()
        mock_anomaly.is_fitted = True
        mock_anomaly.score.return_value = 0.95  # above threshold

        features = pd.DataFrame({"dummy": [1.0]})
        direction, _ = self.ex.compute_signal(
            _make_ohlcv(),
            forecaster=mock_forecaster,
            anomaly_detector=mock_anomaly,
            extra_features=features,
        )
        assert direction == "flat"

    def test_anomaly_below_threshold_allows_signal(self):
        mock_forecaster = MagicMock()
        mock_forecaster.is_fitted = True
        mock_forecaster.predict_proba.return_value = (0.10, 0.10, 0.80)

        mock_anomaly = MagicMock()
        mock_anomaly.is_fitted = True
        mock_anomaly.score.return_value = 0.30  # well below threshold

        features = pd.DataFrame({"dummy": [1.0]})
        direction, _ = self.ex.compute_signal(
            _make_ohlcv(),
            forecaster=mock_forecaster,
            anomaly_detector=mock_anomaly,
            extra_features=features,
        )
        assert direction == "long"


# ---------------------------------------------------------------------------
# run_decision_cycle (paper trading)
# ---------------------------------------------------------------------------

class TestRunDecisionCycle:
    def _executor(self, **kwargs) -> HyperliquidExecutor:
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor(paper_trading=True, **kwargs)
        # Inject mock info that returns a mid-price and account value
        mock_info = MagicMock()
        mock_info.all_mids.return_value = {"ETH": "2000.0"}
        mock_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [],
        }
        ex._info = mock_info
        ex.account_address = "0xABCD"
        return ex

    def test_flat_signal_returns_skipped(self):
        ex = self._executor()
        ohlcv = _make_ohlcv(5)  # too little data → flat
        result = ex.run_decision_cycle("ETH", ohlcv)
        assert result.status == "skipped"
        assert result.side == "none"

    def test_paper_order_on_long_signal(self):
        ex = self._executor()
        # Use a strong uptrend ohlcv so the naive fallback fires
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.status == "paper"
        assert result.side == "buy"
        assert result.size > 0

    def test_paper_order_on_short_signal(self):
        ex = self._executor()
        close = np.linspace(1200, 1000, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.status == "paper"
        assert result.side == "sell"

    def test_circuit_breaker_halts_trading(self):
        ex = self._executor()
        rm = RiskManager(max_drawdown_pct=0.10)
        rm.check_circuit_breaker(10_000.0)  # sets peak

        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        # Pass a capital value that triggers the circuit-breaker (>10% drawdown)
        result = ex.run_decision_cycle(
            "ETH", ohlcv, risk_manager=rm, capital_usd=8_000.0
        )
        assert result.status == "skipped"

    def test_risk_manager_sizes_position(self):
        ex = self._executor()
        rm = RiskManager(max_position_pct=0.05)

        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle(
            "ETH", ohlcv, risk_manager=rm, capital_usd=10_000.0
        )
        # max 5 % of 10k at ~2000/ETH ≈ 0.25 ETH (before vol scaling)
        assert result.size <= 0.25 + 1e-4

    def test_zero_capital_returns_skipped(self):
        ex = self._executor()
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=0.0)
        assert result.status == "skipped"

    def test_no_duplicate_long(self):
        """Should skip when already long in the same direction."""
        ex = self._executor()
        ex._info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "1.0"}}],
        }
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.status == "skipped"

    def test_no_duplicate_short(self):
        """Should skip when already short and receiving another short signal."""
        ex = self._executor()
        ex._info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "-1.0"}}],
        }
        close = np.linspace(1200, 1000, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.status == "skipped"

    def test_close_long_before_open_short(self):
        """When long and receiving a short signal, the long is closed and a short is opened."""
        ex = self._executor()
        ex._info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "1.0"}}],
        }
        close = np.linspace(1200, 1000, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        # Paper close succeeds; new short order should be placed
        assert result.status == "paper"
        assert result.signal_direction == "short"
        assert result.side == "sell"

    def test_close_short_before_open_long(self):
        """When short and receiving a long signal, the short is closed and a long is opened."""
        ex = self._executor()
        ex._info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "-1.0"}}],
        }
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        # Paper close succeeds; new long order should be placed
        assert result.status == "paper"
        assert result.signal_direction == "long"
        assert result.side == "buy"

    def test_reversal_aborted_when_close_rejected(self):
        """Position reversal must abort if the close call is rejected (live mode, no exchange)."""
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor(paper_trading=False)
        mock_info = MagicMock()
        mock_info.all_mids.return_value = {"ETH": "2000.0"}
        mock_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "1.0"}}],
        }
        ex._info = mock_info
        ex.account_address = "0xABCD"

        close = np.linspace(1200, 1000, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        # Close is rejected (no exchange); should not proceed to open opposite side
        assert result.status == "rejected"
        assert result.signal_direction == "short"
        assert 0.0 <= result.signal_strength <= 1.0

    def test_result_has_signal_metadata(self):
        ex = self._executor()
        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.signal_direction in {"long", "short", "flat"}
        assert 0.0 <= result.signal_strength <= 1.0


# ---------------------------------------------------------------------------
# Live mode (no SDK) – rejected path
# ---------------------------------------------------------------------------

class TestLiveModeNoSDK:
    def test_rejected_when_live_but_no_exchange(self):
        with patch("eth_algo_trading.execution.hyperliquid_executor._SDK_AVAILABLE", False):
            ex = HyperliquidExecutor(paper_trading=False)
        mock_info = MagicMock()
        mock_info.all_mids.return_value = {"ETH": "2000.0"}
        mock_info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000.0"},
            "assetPositions": [],
        }
        ex._info = mock_info
        ex.account_address = "0xABCD"

        close = np.linspace(1000, 1200, 20)
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(20)},
            index=idx,
        )
        result = ex.run_decision_cycle("ETH", ohlcv, capital_usd=10_000.0)
        assert result.status == "rejected"
