"""
Inference API for RL trading agent signals and alerts.

Provides a :class:`InferenceEngine` that accepts JSON-serialisable payloads
and returns structured signal and alert objects, plus a :func:`create_flask_app`
factory that exposes these capabilities over HTTP.

JSON payload format (inference request)
----------------------------------------
.. code-block:: json

    {
      "symbol": "ETH/USDT",
      "timeframes": {
        "1h": [
          {"timestamp": "2024-01-01T00:00:00Z",
           "open": 2000, "high": 2020, "low": 1995, "close": 2010, "volume": 1500},
          ...
        ],
        "4h": [...],
        "1d": [...]
      }
    }

JSON response format
---------------------
.. code-block:: json

    {
      "symbol": "ETH/USDT",
      "signal": "long",
      "strength": 0.72,
      "alerts": [
        {"type": "signal", "severity": "info",
         "message": "RL agent emits long signal (strength=0.72)"}
      ],
      "metadata": {
        "q_values": {"flat": -0.1, "long": 0.5, "short": -0.3},
        "is_fitted": true
      }
    }
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from eth_algo_trading.config import RLConfig
from eth_algo_trading.db.hyperparams import HyperparamDB
from eth_algo_trading.models.rl_agent import RLTradingAgent
from eth_algo_trading.strategies.base import Signal

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """A single alert produced during inference."""

    type: str        # 'signal', 'risk', 'anomaly', 'info'
    severity: str    # 'info', 'warning', 'critical'
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """Parsed inference request."""

    symbol: str
    ohlcv_by_tf: Dict[str, pd.DataFrame]


@dataclass
class InferenceResponse:
    """Structured inference response."""

    symbol: str
    signal: str
    strength: float
    alerts: List[Alert]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-serialisable dict."""
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "strength": self.strength,
            "alerts": [asdict(a) for a in self.alerts],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Payload parsing helpers
# ---------------------------------------------------------------------------


def _parse_ohlcv_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of OHLCV dicts (as sent in the JSON payload) to a DataFrame.

    Accepts either an ISO-8601 ``timestamp`` string or a Unix millisecond
    integer.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.RangeIndex(len(df))

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _required = {"open", "high", "low", "close", "volume"}
    missing = _required - set(df.columns)
    if missing:
        raise ValueError(
            f"OHLCV records are missing required columns: {sorted(missing)}"
        )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if df[numeric_cols].isna().values.any():
        raise ValueError(
            "Invalid numeric values in OHLCV records: 'open', 'high', 'low', "
            "'close', and 'volume' must all be valid numbers."
        )

    return df


def _parse_payload(payload: Dict[str, Any]) -> InferenceRequest:
    """Validate and convert a raw JSON payload dict to an InferenceRequest."""
    symbol = str(payload.get("symbol", "ETH/USDT"))
    raw_tfs = payload.get("timeframes", {})
    if not isinstance(raw_tfs, dict):
        raise ValueError("'timeframes' must be a dict mapping label → OHLCV records")

    ohlcv_by_tf: Dict[str, pd.DataFrame] = {}
    for tf_label, records in raw_tfs.items():
        if not isinstance(records, list):
            raise ValueError(f"timeframes['{tf_label}'] must be a list of OHLCV dicts")
        ohlcv_by_tf[str(tf_label)] = _parse_ohlcv_records(records)

    return InferenceRequest(symbol=symbol, ohlcv_by_tf=ohlcv_by_tf)


# ---------------------------------------------------------------------------
# Alert generator
# ---------------------------------------------------------------------------


def _generate_alerts(signal: Signal, symbol: str) -> List[Alert]:
    """Derive actionable alerts from an agent signal."""
    alerts: List[Alert] = []

    direction = signal.direction
    strength = signal.strength

    if direction != "flat":
        severity = "info" if strength < 0.6 else "warning" if strength < 0.85 else "critical"
        alerts.append(
            Alert(
                type="signal",
                severity=severity,
                message=(
                    f"RL agent emits {direction} signal for {symbol} "
                    f"(strength={strength:.2f})"
                ),
                metadata={"direction": direction, "strength": strength},
            )
        )

    q_vals = signal.metadata.get("q_values", {})
    flat_q = q_vals.get("flat", 0.0)
    long_q = q_vals.get("long", 0.0)
    short_q = q_vals.get("short", 0.0)

    # Alert if flat wins by a very small margin (uncertain market)
    if direction == "flat" and abs(long_q - short_q) < 0.05:
        alerts.append(
            Alert(
                type="info",
                severity="info",
                message=f"Market is uncertain for {symbol} — holding flat",
                metadata={"q_values": q_vals},
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """
    Stateless wrapper around :class:`~eth_algo_trading.models.rl_agent.RLTradingAgent`
    that handles JSON payload parsing and alert generation.

    Parameters
    ----------
    agent:
        A pre-constructed (optionally pre-trained) RL agent.  If not supplied,
        a fresh agent with default :class:`~eth_algo_trading.config.RLConfig`
        is created.
    db:
        Optional database for dynamic hyperparameter overrides.
    """

    def __init__(
        self,
        agent: Optional[RLTradingAgent] = None,
        db: Optional[HyperparamDB] = None,
    ) -> None:
        self._db = db
        if agent is not None:
            self._agent = agent
        else:
            config = RLConfig()
            # Let RLTradingAgent handle any DB-based overrides to the config
            self._agent = RLTradingAgent(config=config, db=db)

    def run(self, payload: Dict[str, Any]) -> InferenceResponse:
        """
        Process a JSON payload and return an :class:`InferenceResponse`.

        Parameters
        ----------
        payload:
            Dict matching the documented JSON request schema.

        Returns
        -------
        InferenceResponse
        """
        # Pick up any hyperparameter updates written to the DB since last call.
        if self._db is not None:
            self._agent.refresh_hyperparams()

        request = _parse_payload(payload)

        signal = self._agent.predict(request.ohlcv_by_tf)
        alerts = _generate_alerts(signal, request.symbol)

        return InferenceResponse(
            symbol=request.symbol,
            signal=signal.direction,
            strength=signal.strength,
            alerts=alerts,
            metadata=signal.metadata,
        )

    @property
    def agent(self) -> RLTradingAgent:
        return self._agent


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_flask_app(
    engine: Optional[InferenceEngine] = None,
    db: Optional[HyperparamDB] = None,
) -> Any:
    """
    Create and return a Flask WSGI application exposing the inference API.

    Endpoints
    ---------
    ``POST /inference``
        Accept a JSON body matching the request schema; return an
        :class:`InferenceResponse` as JSON.
    ``GET /health``
        Liveness probe — returns ``{"status": "ok"}``.

    Parameters
    ----------
    engine:
        Pre-built :class:`InferenceEngine`.  A fresh one is created if not
        supplied.
    db:
        Optional hyperparameter database (passed to a freshly created engine).

    Returns
    -------
    Flask app instance, or raises ``ImportError`` if Flask is not installed.
    """
    try:
        from flask import Flask, jsonify, request as flask_request
    except ImportError as exc:
        raise ImportError(
            "flask is required for create_flask_app(). "
            "Install it with: pip install flask"
        ) from exc

    _engine = engine or InferenceEngine(db=db)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():  # type: ignore[return]
        return jsonify({"status": "ok"})

    @app.route("/inference", methods=["POST"])
    def inference():  # type: ignore[return]
        payload = flask_request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        try:
            response = _engine.run(payload)
            return jsonify(response.to_dict()), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception:
            # Log only symbol and timeframe keys to avoid exposing large or
            # potentially sensitive OHLCV data in production logs.
            sanitized: Dict[str, Any] = {}
            if isinstance(payload, dict):
                sanitized["symbol"] = payload.get("symbol")
                tfs = payload.get("timeframes")
                sanitized["timeframes"] = list(tfs.keys()) if isinstance(tfs, dict) else None
            _logger.exception("Unhandled error during inference. Payload metadata: %r", sanitized)
            return jsonify({"error": "Internal inference error"}), 500

    return app
