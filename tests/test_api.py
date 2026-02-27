"""Tests for the inference API."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from eth_algo_trading.api.inference import (
    Alert,
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
    _generate_alerts,
    _parse_ohlcv_records,
    _parse_payload,
)
from eth_algo_trading.config import RLConfig
from eth_algo_trading.db.hyperparams import HyperparamDB
from eth_algo_trading.strategies.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(n: int = 30, seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0, 20, n))
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return [
        {
            "timestamp": timestamps[i].isoformat(),
            "open": float(close[i]),
            "high": float(close[i] + 10),
            "low": float(close[i] - 10),
            "close": float(close[i]),
            "volume": 1000.0,
        }
        for i in range(n)
    ]


def _make_payload(n: int = 30) -> Dict[str, Any]:
    return {
        "symbol": "ETH/USDT",
        "timeframes": {
            "1h": _make_records(n),
            "4h": _make_records(max(5, n // 4)),
        },
    }


# ---------------------------------------------------------------------------
# OHLCV parsing tests
# ---------------------------------------------------------------------------


class TestParseOhlcvRecords:
    def test_basic_parsing(self):
        records = _make_records(10)
        df = _parse_ohlcv_records(records)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "close" in df.columns

    def test_empty_records(self):
        df = _parse_ohlcv_records([])
        assert df.empty

    def test_unix_ms_timestamp(self):
        records = [{"timestamp": 1704067200000, "open": 2000, "high": 2010,
                    "low": 1990, "close": 2005, "volume": 500}]
        df = _parse_ohlcv_records(records)
        assert len(df) == 1

    def test_nan_values_raise_error(self):
        for bad_col in ["open", "high", "low", "close", "volume"]:
            records = _make_records(5)
            records[2][bad_col] = "not-a-number"
            with pytest.raises(ValueError, match="Invalid numeric values"):
                _parse_ohlcv_records(records)

    def test_missing_required_column_raises_error(self):
        for missing_col in ["open", "high", "low", "close", "volume"]:
            records = _make_records(5)
            for r in records:
                del r[missing_col]
            with pytest.raises(ValueError, match="missing required columns"):
                _parse_ohlcv_records(records)


# ---------------------------------------------------------------------------
# Payload parsing tests
# ---------------------------------------------------------------------------


class TestParsePayload:
    def test_valid_payload(self):
        req = _parse_payload(_make_payload())
        assert req.symbol == "ETH/USDT"
        assert "1h" in req.ohlcv_by_tf
        assert "4h" in req.ohlcv_by_tf

    def test_default_symbol(self):
        payload = {"timeframes": {"1h": _make_records(5)}}
        req = _parse_payload(payload)
        assert req.symbol == "ETH/USDT"

    def test_invalid_timeframes_type_raises(self):
        with pytest.raises(ValueError, match="'timeframes' must be a dict"):
            _parse_payload({"symbol": "ETH/USDT", "timeframes": "bad"})

    def test_invalid_records_type_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _parse_payload({"timeframes": {"1h": "not-a-list"}})


# ---------------------------------------------------------------------------
# Alert generation tests
# ---------------------------------------------------------------------------


class TestGenerateAlerts:
    def _make_signal(self, direction: str, strength: float) -> Signal:
        return Signal(
            direction=direction,
            strength=strength,
            strategy="rl_agent",
            metadata={"q_values": {"flat": 0.0, "long": 0.1, "short": -0.1}},
        )

    def test_long_signal_generates_alert(self):
        signal = self._make_signal("long", 0.75)
        alerts = _generate_alerts(signal, "ETH/USDT")
        assert len(alerts) == 1
        assert alerts[0].type == "signal"
        assert "long" in alerts[0].message

    def test_flat_signal_no_signal_alert(self):
        signal = self._make_signal("flat", 0.0)
        alerts = _generate_alerts(signal, "ETH/USDT")
        # Only possible alert is uncertainty info
        assert all(a.type != "signal" for a in alerts)

    def test_high_strength_signal_is_critical(self):
        signal = self._make_signal("long", 0.95)
        alerts = _generate_alerts(signal, "ETH/USDT")
        assert any(a.severity == "critical" for a in alerts)

    def test_low_strength_signal_is_info(self):
        signal = self._make_signal("short", 0.3)
        alerts = _generate_alerts(signal, "ETH/USDT")
        assert any(a.severity == "info" for a in alerts)


# ---------------------------------------------------------------------------
# InferenceEngine tests
# ---------------------------------------------------------------------------


class TestInferenceEngine:
    def test_run_returns_inference_response(self):
        engine = InferenceEngine()
        response = engine.run(_make_payload())
        assert isinstance(response, InferenceResponse)

    def test_response_signal_valid_direction(self):
        engine = InferenceEngine()
        response = engine.run(_make_payload())
        assert response.signal in {"flat", "long", "short"}

    def test_response_strength_in_range(self):
        engine = InferenceEngine()
        response = engine.run(_make_payload())
        assert 0.0 <= response.strength <= 1.0

    def test_response_symbol_matches_payload(self):
        engine = InferenceEngine()
        payload = _make_payload()
        payload["symbol"] = "BTC/USDT"
        response = engine.run(payload)
        assert response.symbol == "BTC/USDT"

    def test_to_dict_is_json_serialisable(self):
        import json

        engine = InferenceEngine()
        response = engine.run(_make_payload())
        d = response.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert "signal" in serialised

    def test_alerts_is_list(self):
        engine = InferenceEngine()
        response = engine.run(_make_payload())
        assert isinstance(response.alerts, list)

    def test_engine_uses_db_hyperparams(self):
        db = HyperparamDB()
        db.set("lookback_bars", 10)
        engine = InferenceEngine(db=db)
        assert engine.agent.config.lookback_bars == 10

    def test_invalid_payload_raises_value_error(self):
        engine = InferenceEngine()
        with pytest.raises(ValueError):
            engine.run({"symbol": "ETH/USDT", "timeframes": "bad"})


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------


class TestCreateFlaskApp:
    def _get_app(self):
        try:
            from eth_algo_trading.api.inference import create_flask_app
        except ImportError:
            pytest.skip("flask not installed")
        return create_flask_app()

    def test_health_endpoint(self):
        app = self._get_app()
        client = app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_inference_endpoint_valid_payload(self):
        import json

        app = self._get_app()
        client = app.test_client()
        payload = _make_payload()
        resp = client.post(
            "/inference",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["signal"] in {"flat", "long", "short"}
        assert "alerts" in data

    def test_inference_endpoint_invalid_json(self):
        app = self._get_app()
        client = app.test_client()
        resp = client.post(
            "/inference",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_inference_endpoint_bad_timeframes(self):
        import json

        app = self._get_app()
        client = app.test_client()
        payload = {"symbol": "ETH/USDT", "timeframes": "bad"}
        resp = client.post(
            "/inference",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 422
