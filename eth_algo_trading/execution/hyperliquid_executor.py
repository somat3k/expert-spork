"""
Hyperliquid execution channel.

Strictly targets the Hyperliquid perpetuals DEX via the
``hyperliquid-python-sdk`` library.  Combines live market data fetched from
Hyperliquid with model-based signal generation (price forecasting, regime
detection, anomaly detection) and volatility-scaled position sizing to form a
complete, real decision-making and order-execution pipeline.

Environment variables (loaded via python-dotenv):
    HL_ACCOUNT_ADDRESS  – 0x-prefixed Ethereum address of the trading account.
    HL_PRIVATE_KEY      – Private key of the account (or an approved API wallet
                          private key authorised via app.hyperliquid.xyz/API).

Typical usage::

    from eth_algo_trading.execution.hyperliquid_executor import HyperliquidExecutor
    from eth_algo_trading.risk.manager import RiskManager

    executor = HyperliquidExecutor(
        account_address="0x…",
        private_key="0x…",
        testnet=True,
        paper_trading=True,
    )
    result = executor.run_decision_cycle(
        coin="ETH",
        ohlcv=ohlcv_df,
        forecaster=fitted_forecaster,
        regime_detector=fitted_regime_detector,
        anomaly_detector=fitted_anomaly_detector,
        risk_manager=RiskManager(),
    )
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import – gracefully degrade when the package is absent so that
# the rest of the codebase can still be imported / tested without it.
# ---------------------------------------------------------------------------

try:
    import eth_account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants as hl_constants

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

if TYPE_CHECKING:
    from eth_algo_trading.models.anomaly_detection import AnomalyDetector
    from eth_algo_trading.models.forecasting import PriceForecaster
    from eth_algo_trading.models.regime_detection import RegimeDetector
    from eth_algo_trading.risk.manager import RiskManager


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HyperliquidOrderResult:
    """Outcome of a single decision-and-execution cycle."""

    coin: str
    side: str                    # 'buy', 'sell', or 'none'
    size: float                  # position size in coin units (e.g. ETH)
    price: float                 # reference mid-price used for sizing
    status: str                  # 'filled', 'rejected', 'paper', 'skipped'
    signal_direction: str        # 'long', 'short', or 'flat'
    signal_strength: float       # 0.0 – 1.0 model confidence
    raw_response: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------


class HyperliquidExecutor:
    """
    Model-driven order executor for the Hyperliquid perpetuals DEX.

    Decision pipeline
    -----------------
    1. **Signal generation** – ``PriceForecaster`` supplies directional
       probabilities; ``RegimeDetector`` provides regime context; and
       ``AnomalyDetector`` flags market anomalies that suppress trading.
    2. **Risk sizing** – ``RiskManager`` computes a volatility-scaled notional
       and converts it to coin units using the live mid-price.
    3. **Circuit breaker** – if portfolio drawdown exceeds the configured
       threshold, no new orders are placed.
    4. **Order routing** – a Hyperliquid market order is placed via
       ``Exchange.market_open`` (or simulated when ``paper_trading=True``).

    Parameters
    ----------
    account_address:
        0x-prefixed on-chain address of the trading account.
    private_key:
        Private key of the account or an authorised API wallet.
    testnet:
        When ``True`` the Hyperliquid testnet endpoint is used.
    paper_trading:
        When ``True`` orders are simulated locally – no network calls to
        ``/exchange`` are made.
    min_prob_threshold:
        Minimum directional probability required to open a position
        (default 0.45).
    max_anomaly_score:
        Anomaly score above which trading is suppressed (default 0.70).
    leverage:
        Leverage applied when opening new positions (default 1 = spot-like).
    """

    def __init__(
        self,
        account_address: str = "",
        private_key: str = "",
        testnet: bool = True,
        paper_trading: bool = True,
        min_prob_threshold: float = 0.45,
        max_anomaly_score: float = 0.70,
        leverage: int = 1,
    ) -> None:
        self.account_address = account_address
        self.paper_trading = paper_trading
        self.min_prob_threshold = min_prob_threshold
        self.max_anomaly_score = max_anomaly_score
        self.leverage = leverage

        self._exchange: Optional[Any] = None
        self._info: Optional[Any] = None

        if not _SDK_AVAILABLE:
            logger.warning(
                "hyperliquid-python-sdk is not installed. "
                "HyperliquidExecutor will operate in offline/paper mode."
            )
            return

        api_url = (
            hl_constants.TESTNET_API_URL if testnet else hl_constants.MAINNET_API_URL
        )

        try:
            self._info = Info(api_url, skip_ws=True)
            if private_key:
                wallet = eth_account.Account.from_key(private_key)
                self._exchange = Exchange(wallet, api_url, account_address=account_address or None)
                logger.info(
                    "HyperliquidExecutor initialised (account=%s, testnet=%s, paper=%s)",
                    account_address,
                    testnet,
                    paper_trading,
                )
        except Exception as exc:
            logger.warning("Failed to initialise Hyperliquid SDK client: %s", exc)
            self._exchange = None
            self._info = None

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def fetch_mid_price(self, coin: str) -> float:
        """
        Return the current Hyperliquid mid-price for *coin*.

        Falls back to ``0.0`` when the SDK is unavailable or the API
        request fails.
        """
        if self._info is None:
            return 0.0
        try:
            mids = self._info.all_mids()
            return float(mids.get(coin, 0.0))
        except Exception as exc:
            logger.warning("fetch_mid_price failed for %s: %s", coin, exc)
            return 0.0

    def fetch_account_value(self) -> float:
        """
        Return the total account equity in USD from Hyperliquid.

        Falls back to ``0.0`` when the SDK is unavailable.
        """
        if self._info is None or not self.account_address:
            return 0.0
        try:
            state = self._info.user_state(self.account_address)
            return float(state.get("marginSummary", {}).get("accountValue", 0.0))
        except Exception as exc:
            logger.warning("fetch_account_value failed: %s", exc)
            return 0.0

    def get_position_size(self, coin: str) -> float:
        """
        Return the current signed position size in coin units.

        Positive = long, negative = short, zero = flat.
        Falls back to ``0.0`` when the SDK is unavailable.
        """
        if self._info is None or not self.account_address:
            return 0.0
        try:
            state = self._info.user_state(self.account_address)
            for pos in state.get("assetPositions", []):
                item = pos.get("position", {})
                if item.get("coin") == coin:
                    return float(item.get("szi", 0.0))
        except Exception as exc:
            logger.warning("get_position_size failed for %s: %s", coin, exc)
        return 0.0

    # ------------------------------------------------------------------
    # Signal generation from model outputs
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        ohlcv: pd.DataFrame,
        forecaster: Optional["PriceForecaster"] = None,
        regime_detector: Optional["RegimeDetector"] = None,
        anomaly_detector: Optional["AnomalyDetector"] = None,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, float]:
        """
        Derive a trading direction and confidence score from model outputs.

        Decision rules
        --------------
        * **Anomaly gate** – if an ``AnomalyDetector`` is provided and its
          score exceeds ``max_anomaly_score``, return ``('flat', 0.0)``
          immediately.
        * **Forecast** – if a fitted ``PriceForecaster`` is available, use
          its ``(prob_down, prob_flat, prob_up)`` output.  A direction is
          chosen when the leading probability exceeds ``min_prob_threshold``.
          Signal strength equals the leading probability.
        * **Regime filter** – a ``'bear'`` regime suppresses long signals; a
          ``'bull'`` regime suppresses short signals.
        * **Fallback** – when no forecaster is provided, the most-recent
          12-bar return is used as a naive directional indicator.

        Returns
        -------
        Tuple[str, float]
            ``(direction, strength)`` where direction is ``'long'``,
            ``'short'``, or ``'flat'``.
        """
        # 1. Anomaly gate
        if anomaly_detector is not None and anomaly_detector.is_fitted:
            try:
                if extra_features is not None and not extra_features.empty:
                    score = anomaly_detector.score(extra_features)
                    if score > self.max_anomaly_score:
                        logger.debug("Anomaly score %.3f exceeds threshold – skipping.", score)
                        return ("flat", 0.0)
            except Exception as exc:
                logger.warning("Anomaly detection failed: %s", exc)

        # 2. Regime context
        regime: Optional[str] = None
        if regime_detector is not None:
            try:
                regime = regime_detector.predict(ohlcv)
            except Exception as exc:
                logger.warning("Regime detection failed: %s", exc)

        # 3. Price forecast
        if forecaster is not None and forecaster.is_fitted:
            try:
                prob_down, prob_flat, prob_up = forecaster.predict_proba(ohlcv, extra_features)
            except Exception as exc:
                logger.warning("Forecaster prediction failed: %s", exc)
                return ("flat", 0.0)

            if prob_up >= prob_down and prob_up >= prob_flat and prob_up >= self.min_prob_threshold:
                direction, strength = "long", prob_up
            elif prob_down >= prob_up and prob_down >= prob_flat and prob_down >= self.min_prob_threshold:
                direction, strength = "short", prob_down
            else:
                return ("flat", 0.0)

            # Regime filter
            if regime == "bear" and direction == "long":
                return ("flat", 0.0)
            if regime == "bull" and direction == "short":
                return ("flat", 0.0)

            return (direction, round(strength, 4))

        # 4. Naive fallback using recent return
        if len(ohlcv) < 13:
            return ("flat", 0.0)

        ret_12 = float(ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-13] - 1)
        if ret_12 > 0.005:
            direction, strength = "long", min(1.0, abs(ret_12) * 20)
        elif ret_12 < -0.005:
            direction, strength = "short", min(1.0, abs(ret_12) * 20)
        else:
            return ("flat", 0.0)

        if regime == "bear" and direction == "long":
            return ("flat", 0.0)
        if regime == "bull" and direction == "short":
            return ("flat", 0.0)

        return (direction, round(strength, 4))

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        mid_price: float,
    ) -> HyperliquidOrderResult:
        """Submit a market order to Hyperliquid or simulate it in paper mode."""
        side = "buy" if is_buy else "sell"

        if self.paper_trading:
            logger.info(
                "[PAPER] %s %s coin=%s size=%.6f price=%.2f",
                "BUY" if is_buy else "SELL",
                "paper",
                coin,
                size,
                mid_price,
            )
            return HyperliquidOrderResult(
                coin=coin,
                side=side,
                size=size,
                price=mid_price,
                status="paper",
                signal_direction="long" if is_buy else "short",
                signal_strength=0.0,
                raw_response={},
            )

        if self._exchange is None:
            logger.error("Exchange client is not initialised; cannot place live order.")
            return HyperliquidOrderResult(
                coin=coin, side=side, size=size, price=mid_price,
                status="rejected", signal_direction="long" if is_buy else "short",
                signal_strength=0.0,
            )

        try:
            if self.leverage != 1:
                try:
                    self._exchange.update_leverage(self.leverage, coin)
                except Exception as lev_exc:  # SDK may raise various errors; log and proceed
                    logger.warning(
                        "Failed to set leverage %s for %s: %s", self.leverage, coin, lev_exc
                    )
            response = self._exchange.market_open(coin, is_buy, size)
            logger.info("Order placed: %s", response)
            return HyperliquidOrderResult(
                coin=coin,
                side=side,
                size=size,
                price=mid_price,
                status="filled",
                signal_direction="long" if is_buy else "short",
                signal_strength=0.0,
                raw_response=response if isinstance(response, dict) else {},
            )
        except Exception as exc:
            logger.error("Order placement failed for %s: %s", coin, exc)
            return HyperliquidOrderResult(
                coin=coin, side=side, size=size, price=mid_price,
                status="rejected", signal_direction="long" if is_buy else "short",
                signal_strength=0.0,
            )

    def _close_position(
        self,
        coin: str,
        mid_price: float,
        current_size: float,
    ) -> HyperliquidOrderResult:
        """Close an existing open position."""
        is_buy = current_size < 0  # buy to close a short
        side = "buy" if is_buy else "sell"
        abs_size = abs(current_size)

        if self.paper_trading:
            logger.info("[PAPER] CLOSE position coin=%s size=%.6f", coin, abs_size)
            return HyperliquidOrderResult(
                coin=coin, side=side, size=abs_size, price=mid_price,
                status="paper", signal_direction="flat", signal_strength=0.0,
            )

        if self._exchange is None:
            return HyperliquidOrderResult(
                coin=coin, side=side, size=abs_size, price=mid_price,
                status="rejected", signal_direction="flat", signal_strength=0.0,
            )

        try:
            response = self._exchange.market_close(coin, sz=abs_size)
            return HyperliquidOrderResult(
                coin=coin, side=side, size=abs_size, price=mid_price,
                status="filled", signal_direction="flat", signal_strength=0.0,
                raw_response=response if isinstance(response, dict) else {},
            )
        except Exception as exc:
            logger.error("Position close failed for %s: %s", coin, exc)
            return HyperliquidOrderResult(
                coin=coin, side=side, size=abs_size, price=mid_price,
                status="rejected", signal_direction="flat", signal_strength=0.0,
            )

    # ------------------------------------------------------------------
    # Main decision-and-execution cycle
    # ------------------------------------------------------------------

    def run_decision_cycle(
        self,
        coin: str,
        ohlcv: pd.DataFrame,
        forecaster: Optional["PriceForecaster"] = None,
        regime_detector: Optional["RegimeDetector"] = None,
        anomaly_detector: Optional["AnomalyDetector"] = None,
        risk_manager: Optional["RiskManager"] = None,
        extra_features: Optional[pd.DataFrame] = None,
        capital_usd: Optional[float] = None,
    ) -> HyperliquidOrderResult:
        """
        Run a full model-driven decision-and-execution cycle on Hyperliquid.

        Steps
        -----
        1. Derive direction + confidence from model outputs via
           :meth:`compute_signal`.
        2. Fetch the live mid-price from Hyperliquid (or use OHLCV close as
           fallback).
        3. Determine current account equity (live or provided *capital_usd*).
        4. Run the circuit-breaker; skip if triggered.
        5. Compute volatility-scaled position size via :class:`RiskManager`.
        6. Check the current open position to avoid directional duplication.
        7. If the signal is contradictory to the current position, close first.
        8. Execute the market order via :meth:`_execute_order`.

        Parameters
        ----------
        coin:
            Hyperliquid coin name (e.g. ``'ETH'``).
        ohlcv:
            OHLCV DataFrame aligned to UTC timestamps.
        forecaster:
            Optional fitted :class:`~eth_algo_trading.models.forecasting.PriceForecaster`.
        regime_detector:
            Optional :class:`~eth_algo_trading.models.regime_detection.RegimeDetector`.
        anomaly_detector:
            Optional fitted :class:`~eth_algo_trading.models.anomaly_detection.AnomalyDetector`.
        risk_manager:
            Optional :class:`~eth_algo_trading.risk.manager.RiskManager` used for
            position sizing and circuit-breaker logic.
        extra_features:
            Optional supplementary feature DataFrame passed to models.
        capital_usd:
            Override for available capital (USD).  When *None*, fetched live
            from Hyperliquid.

        Returns
        -------
        HyperliquidOrderResult
        """
        # 1. Signal generation
        direction, strength = self.compute_signal(
            ohlcv, forecaster, regime_detector, anomaly_detector, extra_features
        )

        if direction == "flat":
            return HyperliquidOrderResult(
                coin=coin,
                side="none",
                size=0.0,
                price=0.0,
                status="skipped",
                signal_direction="flat",
                signal_strength=strength,
            )

        # 2. Mid-price
        mid_price = self.fetch_mid_price(coin)
        if mid_price <= 0.0:
            # Fallback to most recent OHLCV close
            if not ohlcv.empty:
                mid_price = float(ohlcv["close"].iloc[-1])
            if mid_price <= 0.0:
                logger.warning("Cannot determine mid-price for %s; skipping.", coin)
                return HyperliquidOrderResult(
                    coin=coin, side="none", size=0.0, price=0.0,
                    status="skipped", signal_direction=direction, signal_strength=strength,
                )

        # 3. Capital
        capital = capital_usd if capital_usd is not None else self.fetch_account_value()
        if capital <= 0.0:
            logger.warning("Account value is zero or unavailable; skipping cycle.")
            return HyperliquidOrderResult(
                coin=coin, side="none", size=0.0, price=mid_price,
                status="skipped", signal_direction=direction, signal_strength=strength,
            )

        # 4. Circuit breaker
        if risk_manager is not None and risk_manager.check_circuit_breaker(capital):
            logger.warning("Circuit breaker triggered (capital=%.2f); halting trading.", capital)
            return HyperliquidOrderResult(
                coin=coin, side="none", size=0.0, price=mid_price,
                status="skipped", signal_direction=direction, signal_strength=strength,
            )

        # 5. Position sizing
        size_eth: float
        if risk_manager is not None:
            sizing = risk_manager.compute_position_size(
                capital_usd=capital,
                entry_price=mid_price,
                signal_strength=strength,
                ohlcv=ohlcv,
            )
            size_eth = sizing.size_eth
        else:
            # Default: 1 % of capital at current price
            size_eth = round((capital * 0.01) / mid_price, 6)

        if size_eth <= 0.0:
            return HyperliquidOrderResult(
                coin=coin, side="none", size=0.0, price=mid_price,
                status="skipped", signal_direction=direction, signal_strength=strength,
            )

        # 6. Current position
        current_pos = self.get_position_size(coin)
        is_long_signal = direction == "long"

        # 7. Close opposing position before reversing
        if current_pos > 0 and not is_long_signal:
            logger.info("Closing long position before shorting %s (size=%.6f).", coin, current_pos)
            close_result = self._close_position(coin, mid_price, current_pos)
            if close_result.status == "rejected":
                logger.error(
                    "Failed to close long position for %s before opening short; "
                    "aborting reversal to avoid increasing exposure.",
                    coin,
                )
                close_result.signal_direction = direction
                close_result.signal_strength = strength
                return close_result
        elif current_pos < 0 and is_long_signal:
            logger.info("Closing short position before going long %s (size=%.6f).", coin, abs(current_pos))
            close_result = self._close_position(coin, mid_price, current_pos)
            if close_result.status == "rejected":
                logger.error(
                    "Failed to close short position for %s before opening long; "
                    "aborting reversal to avoid increasing exposure.",
                    coin,
                )
                close_result.signal_direction = direction
                close_result.signal_strength = strength
                return close_result
        elif (current_pos > 0 and is_long_signal) or (current_pos < 0 and not is_long_signal):
            # Already positioned in the same direction – skip to avoid duplication
            logger.debug("Already positioned in %s direction for %s; skipping.", direction, coin)
            return HyperliquidOrderResult(
                coin=coin, side="none", size=0.0, price=mid_price,
                status="skipped", signal_direction=direction, signal_strength=strength,
            )

        # 8. Execute
        result = self._execute_order(coin, is_long_signal, size_eth, mid_price)
        result.signal_direction = direction
        result.signal_strength = strength
        return result
