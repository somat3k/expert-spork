"""Tests for risk manager."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.risk.manager import RiskManager, PositionSizeResult


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 2000.0 + np.cumsum(rng.normal(0, 20, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 5, "low": close - 5, "close": close, "volume": np.ones(n) * 500},
        index=idx,
    )


class TestRiskManager:
    def test_position_size_basic(self):
        rm = RiskManager(max_position_pct=0.10)
        result = rm.compute_position_size(
            capital_usd=10_000,
            entry_price=2_000,
            signal_strength=1.0,
        )
        assert isinstance(result, PositionSizeResult)
        assert result.size_usd <= 1_000  # 10 % of 10k

    def test_position_size_scales_with_signal(self):
        rm = RiskManager(max_position_pct=0.10)
        r1 = rm.compute_position_size(10_000, 2_000, signal_strength=0.5)
        r2 = rm.compute_position_size(10_000, 2_000, signal_strength=1.0)
        assert r1.size_usd < r2.size_usd

    def test_position_size_with_ohlcv(self):
        rm = RiskManager()
        ohlcv = _make_ohlcv()
        result = rm.compute_position_size(10_000, 2_000, 1.0, ohlcv)
        assert result.size_usd > 0

    def test_invalid_capital(self):
        rm = RiskManager()
        result = rm.compute_position_size(0, 2_000, 0.5)
        assert result.size_usd == 0.0

    def test_invalid_price(self):
        rm = RiskManager()
        result = rm.compute_position_size(10_000, 0, 0.5)
        assert result.size_usd == 0.0

    def test_stop_and_take_profit(self):
        rm = RiskManager(stop_loss_pct=0.03, take_profit_pct=0.06)
        result = rm.compute_position_size(10_000, 2_000, 1.0)
        assert result.stop_loss_price == pytest.approx(2_000 * 0.97, rel=1e-4)
        assert result.take_profit_price == pytest.approx(2_000 * 1.06, rel=1e-4)

    def test_circuit_breaker_not_triggered(self):
        rm = RiskManager(max_drawdown_pct=0.15)
        assert not rm.check_circuit_breaker(10_000)
        assert not rm.check_circuit_breaker(9_000)  # 10% drawdown - within limit

    def test_circuit_breaker_triggered(self):
        rm = RiskManager(max_drawdown_pct=0.15)
        rm.check_circuit_breaker(10_000)      # sets peak
        assert rm.check_circuit_breaker(8_400)  # 16% drawdown - exceeds limit

    def test_circuit_breaker_resets_on_new_high(self):
        rm = RiskManager(max_drawdown_pct=0.15)
        rm.check_circuit_breaker(10_000)
        rm.check_circuit_breaker(12_000)  # new high
        assert not rm.check_circuit_breaker(11_000)  # 8% from new high
