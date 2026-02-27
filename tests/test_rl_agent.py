"""Tests for the RL trading agent and hyperparameter database."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.config import RLConfig
from eth_algo_trading.db.hyperparams import HyperparamDB
from eth_algo_trading.models.rl_agent import (
    RLTradingAgent,
    _build_state,
    _state_dim,
)
from eth_algo_trading.strategies.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0, 20, n))
    high = close + rng.uniform(5, 30, n)
    low = close - rng.uniform(5, 30, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": rng.uniform(500, 2000, n)},
        index=idx,
    )


def _default_ohlcv_by_tf(n: int = 120) -> dict:
    return {"1h": _make_ohlcv(n, 0), "4h": _make_ohlcv(n // 4 + 1, 1), "1d": _make_ohlcv(max(10, n // 24 + 1), 2)}


# ---------------------------------------------------------------------------
# HyperparamDB tests
# ---------------------------------------------------------------------------


class TestHyperparamDB:
    def test_set_and_get(self):
        db = HyperparamDB()
        db.set("gamma", 0.95)
        assert db.get("gamma") == pytest.approx(0.95)

    def test_get_default_when_missing(self):
        db = HyperparamDB()
        assert db.get("nonexistent", default=42) == 42

    def test_update_overwrites(self):
        db = HyperparamDB()
        db.set("lr", 1e-3)
        db.set("lr", 5e-4)
        assert db.get("lr") == pytest.approx(5e-4)

    def test_load_all_returns_dict(self):
        db = HyperparamDB()
        db.set("alpha", 0.1)
        db.set("beta", 0.2)
        result = db.load_all()
        assert result["alpha"] == pytest.approx(0.1)
        assert result["beta"] == pytest.approx(0.2)

    def test_delete_removes_key(self):
        db = HyperparamDB()
        db.set("key1", "value1")
        db.delete("key1")
        assert db.get("key1") is None

    def test_save_and_load_blob(self):
        db = HyperparamDB()
        data = b"\x00\x01\x02\x03"
        db.save_blob("weights", data)
        loaded = db.load_blob("weights")
        assert loaded == data

    def test_load_blob_missing_returns_none(self):
        db = HyperparamDB()
        assert db.load_blob("no_such_key") is None

    def test_context_manager(self):
        with HyperparamDB() as db:
            db.set("x", 99)
            assert db.get("x") == 99

    def test_stores_list_value(self):
        db = HyperparamDB()
        db.set("timeframes", ["1h", "4h", "1d"])
        assert db.get("timeframes") == ["1h", "4h", "1d"]


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestBuildState:
    def test_output_shape(self):
        ohlcv_by_tf = _default_ohlcv_by_tf()
        cfg = RLConfig(timeframes=["1h", "4h", "1d"], lookback_bars=20)
        state = _build_state(ohlcv_by_tf, cfg.lookback_bars)
        expected_dim = _state_dim(len(cfg.timeframes), cfg.lookback_bars)
        assert state.shape == (expected_dim,)

    def test_output_dtype(self):
        state = _build_state(_default_ohlcv_by_tf(), lookback=20)
        assert state.dtype == np.float32

    def test_short_data_zero_padded(self):
        """DataFrames shorter than lookback should produce a valid state."""
        short = _make_ohlcv(5)
        state = _build_state({"1h": short}, lookback=20)
        assert state.shape == (_state_dim(1, 20),)
        assert not np.any(np.isnan(state))

    def test_empty_dataframe_gives_zeros(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        state = _build_state({"1h": empty}, lookback=10)
        assert np.all(state == 0)


# ---------------------------------------------------------------------------
# RLTradingAgent tests
# ---------------------------------------------------------------------------


class TestRLTradingAgent:
    def test_predict_returns_signal(self):
        agent = RLTradingAgent(config=RLConfig(lookback_bars=20))
        signal = agent.predict(_default_ohlcv_by_tf())
        assert isinstance(signal, Signal)
        assert signal.strategy == "rl_agent"
        assert signal.direction in {"flat", "long", "short"}
        assert 0.0 <= signal.strength <= 1.0

    def test_predict_metadata_has_q_values(self):
        agent = RLTradingAgent(config=RLConfig(lookback_bars=20))
        signal = agent.predict(_default_ohlcv_by_tf())
        assert "q_values" in signal.metadata
        q = signal.metadata["q_values"]
        assert "flat" in q and "long" in q and "short" in q

    def test_is_fitted_starts_false(self):
        agent = RLTradingAgent()
        assert not agent.is_fitted

    def test_fit_sets_fitted(self):
        agent = RLTradingAgent(config=RLConfig(timeframes=["1h"], lookback_bars=10, batch_size=4))
        ohlcv_by_tf = {"1h": _make_ohlcv(30)}
        agent.fit(ohlcv_by_tf, n_episodes=1)
        assert agent.is_fitted

    def test_fit_with_insufficient_data_stays_unfitted(self):
        agent = RLTradingAgent(config=RLConfig(timeframes=["1h"], lookback_bars=50))
        # Only 10 bars — less than lookback
        small = _make_ohlcv(10)
        agent.fit({"1h": small})
        assert not agent.is_fitted

    def test_hyperparams_loaded_from_db(self):
        db = HyperparamDB()
        db.set("gamma", 0.88)
        agent = RLTradingAgent(config=RLConfig(), db=db)
        assert agent.config.gamma == pytest.approx(0.88)

    def test_refresh_hyperparams_picks_up_db_change(self):
        db = HyperparamDB()
        agent = RLTradingAgent(config=RLConfig(gamma=0.99), db=db)
        db.set("gamma", 0.77)
        agent.refresh_hyperparams()
        assert agent.config.gamma == pytest.approx(0.77)

    def test_save_and_load_checkpoint(self):
        db = HyperparamDB()
        agent1 = RLTradingAgent(config=RLConfig(timeframes=["1h"], lookback_bars=10, batch_size=4))
        ohlcv_by_tf = {"1h": _make_ohlcv(30)}
        agent1.fit(ohlcv_by_tf, n_episodes=1)
        agent1.save_checkpoint(db)

        agent2 = RLTradingAgent(config=RLConfig(timeframes=["1h"], lookback_bars=10), db=db)
        agent2.load_checkpoint(db)
        assert agent2.is_fitted

    def test_select_action_greedy(self):
        agent = RLTradingAgent(config=RLConfig(lookback_bars=10))
        state = np.zeros(agent._online_net.input_dim, dtype=np.float32)
        action = agent.select_action(state, greedy=True)
        assert action in {0, 1, 2}

    def test_store_and_update_experience(self):
        agent = RLTradingAgent(config=RLConfig(lookback_bars=10, batch_size=4, replay_capacity=100))
        dim = agent._online_net.input_dim
        for _ in range(10):
            s = np.random.rand(dim).astype(np.float32)
            ns = np.random.rand(dim).astype(np.float32)
            agent.store_experience(s, 0, 0.1, ns, False)
        loss = agent.update()
        # loss should be a non-negative float (0.0 if torch not available)
        assert loss >= 0.0
