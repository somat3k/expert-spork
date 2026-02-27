"""Tests for strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.strategies.base import Signal
from eth_algo_trading.strategies.scalping import ScalpingStrategy
from eth_algo_trading.strategies.arbitrage import ArbitrageStrategy
from eth_algo_trading.strategies.trend_following import TrendFollowingStrategy
from eth_algo_trading.strategies.sentiment import SentimentStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create synthetic OHLCV data with an optional trend."""
    rng = np.random.default_rng(42)
    close = 2000.0 * np.cumprod(1 + trend + rng.normal(0, 0.005, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignal:
    def test_valid_long(self):
        s = Signal(direction="long", strength=0.8, strategy="test")
        assert s.direction == "long"
        assert s.strength == 0.8

    def test_valid_flat(self):
        s = Signal(direction="flat", strength=0.0, strategy="test")
        assert s.direction == "flat"

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(direction="up", strength=0.5, strategy="test")

    def test_invalid_strength(self):
        with pytest.raises(ValueError, match="strength"):
            Signal(direction="long", strength=1.5, strategy="test")


# ---------------------------------------------------------------------------
# Scalping strategy
# ---------------------------------------------------------------------------

class TestScalpingStrategy:
    def test_returns_signal(self):
        strat = ScalpingStrategy()
        ohlcv = _make_ohlcv(50)
        signal = strat.generate_signal(ohlcv)
        assert isinstance(signal, Signal)
        assert signal.strategy == "scalping"

    def test_flat_when_insufficient_data(self):
        strat = ScalpingStrategy(rsi_period=14)
        ohlcv = _make_ohlcv(5)
        signal = strat.generate_signal(ohlcv)
        assert signal.direction == "flat"

    def test_oversold_triggers_long(self):
        strat = ScalpingStrategy(rsi_period=5, oversold_threshold=50.0)
        # Create a strongly downtrending series to push RSI below 50
        n = 30
        close = np.linspace(3000, 1000, n)
        high = close + 10
        low = close - 10
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n) * 1000},
            index=idx,
        )
        signal = strat.generate_signal(ohlcv)
        assert signal.direction == "long"

    def test_overbought_triggers_short(self):
        strat = ScalpingStrategy(rsi_period=5, overbought_threshold=50.0)
        n = 30
        close = np.linspace(1000, 3000, n)
        high = close + 10
        low = close - 10
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n) * 1000},
            index=idx,
        )
        signal = strat.generate_signal(ohlcv)
        assert signal.direction == "short"

    def test_strength_in_range(self):
        strat = ScalpingStrategy()
        ohlcv = _make_ohlcv(100)
        signal = strat.generate_signal(ohlcv)
        assert 0.0 <= signal.strength <= 1.0


# ---------------------------------------------------------------------------
# Arbitrage strategy
# ---------------------------------------------------------------------------

class TestArbitrageStrategy:
    def test_flat_without_features(self):
        strat = ArbitrageStrategy()
        signal = strat.generate_signal(_make_ohlcv())
        assert signal.direction == "flat"

    def test_long_when_venue_a_cheaper(self):
        strat = ArbitrageStrategy(min_spread_pct=0.001)
        features = pd.DataFrame({"price_venue_a": [1990.0], "price_venue_b": [2010.0]})
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "long"
        assert signal.strength > 0

    def test_short_when_venue_b_cheaper(self):
        strat = ArbitrageStrategy(min_spread_pct=0.001)
        features = pd.DataFrame({"price_venue_a": [2010.0], "price_venue_b": [1990.0]})
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "short"

    def test_flat_below_min_spread(self):
        strat = ArbitrageStrategy(min_spread_pct=0.05)
        features = pd.DataFrame({"price_venue_a": [2000.0], "price_venue_b": [2001.0]})
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "flat"

    def test_compute_spread(self):
        strat = ArbitrageStrategy()
        spread = strat.compute_spread(100.0, 110.0)
        assert abs(spread - (10 / 105)) < 1e-9


# ---------------------------------------------------------------------------
# Trend-following strategy
# ---------------------------------------------------------------------------

class TestTrendFollowingStrategy:
    def test_flat_when_insufficient_data(self):
        strat = TrendFollowingStrategy(fast_period=20, slow_period=50)
        ohlcv = _make_ohlcv(30)
        signal = strat.generate_signal(ohlcv)
        assert signal.direction == "flat"

    def test_returns_signal_on_enough_data(self):
        strat = TrendFollowingStrategy(fast_period=5, slow_period=10, adx_threshold=0.0)
        ohlcv = _make_ohlcv(100)
        signal = strat.generate_signal(ohlcv)
        assert isinstance(signal, Signal)
        assert signal.direction in {"long", "short", "flat"}

    def test_regime_sideways_forces_flat(self):
        strat = TrendFollowingStrategy(fast_period=5, slow_period=10, adx_threshold=0.0)
        ohlcv = _make_ohlcv(100)
        features = pd.DataFrame({"regime": ["sideways"]}, index=[ohlcv.index[-1]])
        signal = strat.generate_signal(ohlcv, features)
        assert signal.direction == "flat"


# ---------------------------------------------------------------------------
# Sentiment strategy
# ---------------------------------------------------------------------------

class TestSentimentStrategy:
    def test_flat_without_features(self):
        strat = SentimentStrategy()
        signal = strat.generate_signal(_make_ohlcv())
        assert signal.direction == "flat"

    def test_long_on_positive_sentiment(self):
        strat = SentimentStrategy(sentiment_threshold=0.2)
        features = pd.DataFrame({
            "sentiment_score": [0.8],
            "net_exchange_inflow": [-100.0],
            "whale_anomaly_score": [0.0],
        })
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "long"

    def test_short_on_negative_sentiment(self):
        strat = SentimentStrategy(sentiment_threshold=0.2)
        features = pd.DataFrame({
            "sentiment_score": [-0.8],
            "net_exchange_inflow": [100.0],
            "whale_anomaly_score": [0.0],
        })
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "short"

    def test_flat_below_threshold(self):
        strat = SentimentStrategy(sentiment_threshold=0.5)
        features = pd.DataFrame({
            "sentiment_score": [0.1],
            "net_exchange_inflow": [0.0],
            "whale_anomaly_score": [0.0],
        })
        signal = strat.generate_signal(_make_ohlcv(), features)
        assert signal.direction == "flat"
